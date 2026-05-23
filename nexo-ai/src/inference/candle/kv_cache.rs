use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::api::model_traits::KvCacheable;
use crate::api::types::LayerKvSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedKvLayerMetadata {
    pub layer_idx: usize,
    pub is_sliding: bool,
    pub offset: usize,
    pub current_seq_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedKvCacheMetadata {
    pub session_id: String,
    pub model_name: String,
    pub processed_tokens: Vec<u32>,
    pub layer_count: usize,
    #[serde(default)]
    pub layers: Vec<PersistedKvLayerMetadata>,
    pub created_at: String,
    pub last_accessed: String,
}

/// Persist the current in-memory KV cache for a model to disk.
///
/// Returns `Ok(true)` when cache state was written and `Ok(false)` when the
/// model has no active session or no cached tokens to persist.
pub fn save_model_cache_to_disk(
    dir: &Path,
    model_name: &str,
    model: &dyn KvCacheable,
) -> Result<bool> {
    let Some(session_id) = model.current_session_id() else {
        tracing::debug!("KV cache: no current session to save");
        return Ok(false);
    };

    if model.kv_cache_seq_len() == 0 {
        tracing::debug!("KV cache: empty cache for session {session_id}, skipping save");
        return Ok(false);
    }

    let snapshots = model.save_kv_cache()?;
    save_snapshots_to_disk(
        dir,
        session_id,
        model_name,
        &snapshots,
        model.processed_tokens(),
    )?;
    Ok(true)
}

/// Restore a session's KV cache from disk into the given model.
pub fn load_model_cache_from_disk(
    dir: &Path,
    session_id: &str,
    model: &mut dyn KvCacheable,
) -> Result<Option<PersistedKvCacheMetadata>> {
    let device = model.device().clone();
    let dtype = model.dtype();
    let Some((snapshots, metadata)) = load_snapshots_from_disk(dir, session_id, &device, dtype)?
    else {
        return Ok(None);
    };

    model.clear_kv_cache();
    model.restore_kv_cache(&snapshots)?;
    model.set_session_state(
        Some(metadata.session_id.clone()),
        metadata.processed_tokens.clone(),
    );

    Ok(Some(metadata))
}

/// Delete a persisted session from disk.
pub fn delete_model_cache_from_disk(dir: &Path, session_id: &str) -> Result<()> {
    let safetensors_path = dir.join(format!("{session_id}.safetensors"));
    let metadata_path = dir.join(format!("{session_id}.json"));

    if safetensors_path.exists() {
        std::fs::remove_file(&safetensors_path)?;
    }
    if metadata_path.exists() {
        std::fs::remove_file(&metadata_path)?;
    }

    Ok(())
}

/// Expire persisted sessions by age and entry count.
pub fn expire_model_caches_on_disk(
    dir: &Path,
    max_entries: usize,
    max_age: Duration,
) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }

    let mut entries: Vec<(String, PersistedKvCacheMetadata)> = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json")
            && let Ok(json) = std::fs::read_to_string(&path)
            && let Ok(metadata) = serde_json::from_str::<PersistedKvCacheMetadata>(&json)
        {
            entries.push((metadata.session_id.clone(), metadata));
        }
    }

    entries.sort_by(|left, right| left.1.last_accessed.cmp(&right.1.last_accessed));

    let now = epoch_seconds_string();
    let mut deleted = 0;

    for (session_id, metadata) in &entries {
        let is_expired = is_older_than(&metadata.last_accessed, &now, max_age);
        let over_capacity = entries.len() - deleted > max_entries;

        if is_expired || over_capacity {
            if let Err(error) = delete_model_cache_from_disk(dir, session_id) {
                tracing::warn!("Failed to delete cache for session {session_id}: {error}");
            } else {
                deleted += 1;
            }
        }
    }

    if deleted > 0 {
        tracing::debug!("Expired {deleted} old KV cache entries from disk");
    }

    Ok(deleted)
}

fn save_snapshots_to_disk(
    dir: &Path,
    session_id: &str,
    model_name: &str,
    snapshots: &[LayerKvSnapshot],
    processed_tokens: &[u32],
) -> Result<()> {
    let start = std::time::Instant::now();
    std::fs::create_dir_all(dir)?;

    let safetensors_path = dir.join(format!("{session_id}.safetensors"));
    let metadata_path = dir.join(format!("{session_id}.json"));

    let mut tensor_map: HashMap<String, Tensor> = HashMap::new();
    for snapshot in snapshots {
        if let Some(key) = &snapshot.k_data {
            tensor_map.insert(
                format!("layer_{}_k", snapshot.layer_idx),
                key.to_device(&Device::Cpu)?,
            );
        }
        if let Some(value) = &snapshot.v_data {
            tensor_map.insert(
                format!("layer_{}_v", snapshot.layer_idx),
                value.to_device(&Device::Cpu)?,
            );
        }
    }

    candle_core::safetensors::save(&tensor_map, &safetensors_path)?;

    let now = epoch_seconds_string();
    let metadata = PersistedKvCacheMetadata {
        session_id: session_id.to_string(),
        model_name: model_name.to_string(),
        processed_tokens: processed_tokens.to_vec(),
        layer_count: snapshots.len(),
        layers: snapshots
            .iter()
            .map(|snapshot| PersistedKvLayerMetadata {
                layer_idx: snapshot.layer_idx,
                is_sliding: snapshot.is_sliding,
                offset: snapshot.offset,
                current_seq_len: snapshot.current_seq_len,
            })
            .collect(),
        created_at: now.clone(),
        last_accessed: now,
    };
    let json = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(&metadata_path, json)?;

    tracing::debug!(
        "KV cache saved to disk for session {session_id} ({} layers, {:.1}ms)",
        snapshots.len(),
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(())
}

fn load_snapshots_from_disk(
    dir: &Path,
    session_id: &str,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Vec<LayerKvSnapshot>, PersistedKvCacheMetadata)>> {
    let safetensors_path = dir.join(format!("{session_id}.safetensors"));
    let metadata_path = dir.join(format!("{session_id}.json"));

    if !safetensors_path.exists() || !metadata_path.exists() {
        return Ok(None);
    }

    let start = std::time::Instant::now();

    let json = std::fs::read_to_string(&metadata_path)?;
    let mut metadata: PersistedKvCacheMetadata = serde_json::from_str(&json)?;
    metadata.last_accessed = epoch_seconds_string();
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    let tensors = candle_core::safetensors::load(&safetensors_path, device)?;

    let layers = if metadata.layers.is_empty() {
        (0..metadata.layer_count)
            .map(|layer_idx| PersistedKvLayerMetadata {
                layer_idx,
                is_sliding: false,
                offset: 0,
                current_seq_len: 0,
            })
            .collect::<Vec<_>>()
    } else {
        metadata.layers.clone()
    };

    let mut snapshots = Vec::with_capacity(layers.len());
    for layer in layers {
        let key_name = format!("layer_{}_k", layer.layer_idx);
        let value_name = format!("layer_{}_v", layer.layer_idx);

        let k_data = tensors
            .get(&key_name)
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;
        let v_data = tensors
            .get(&value_name)
            .map(|tensor| tensor.to_dtype(dtype))
            .transpose()?;

        let inferred_seq_len = k_data
            .as_ref()
            .and_then(|tensor| tensor.dim(2).ok())
            .unwrap_or(0);
        let current_seq_len = if layer.current_seq_len == 0 {
            inferred_seq_len
        } else {
            layer.current_seq_len
        };

        snapshots.push(LayerKvSnapshot {
            layer_idx: layer.layer_idx,
            is_sliding: layer.is_sliding,
            k_data,
            v_data,
            offset: layer.offset,
            current_seq_len,
        });
    }

    tracing::debug!(
        "KV cache loaded from disk for session {session_id} ({} layers, {:.1}ms)",
        snapshots.len(),
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(Some((snapshots, metadata)))
}

fn epoch_seconds_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn is_older_than(timestamp_str: &str, now_str: &str, max_age: Duration) -> bool {
    let Ok(timestamp) = timestamp_str.parse::<u64>() else {
        return true;
    };
    let Ok(now) = now_str.parse::<u64>() else {
        return false;
    };
    now.saturating_sub(timestamp) > max_age.as_secs()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn epoch_seconds_string_is_numeric() {
        assert!(epoch_seconds_string().parse::<u64>().is_ok());
    }

    #[test]
    fn is_older_than_works() {
        assert!(is_older_than("1000", "5000", Duration::from_secs(3600)));
        assert!(!is_older_than("4000", "5000", Duration::from_secs(3600)));
    }
}