use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use nexo_ai::shared::types::LayerKvSnapshot;

use super::types::CacheMetadata;

/// Save KV cache snapshots to disk as safetensors + JSON metadata.
pub fn save_to_disk(
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

    // Collect tensors, moving to CPU for serialization
    let mut tensor_map: HashMap<String, Tensor> = HashMap::new();

    for snap in snapshots {
        if let Some(k) = &snap.k_data {
            let name = format!("layer_{}_k", snap.layer_idx);
            tensor_map.insert(name, k.to_device(&Device::Cpu)?);
        }
        if let Some(v) = &snap.v_data {
            let name = format!("layer_{}_v", snap.layer_idx);
            tensor_map.insert(name, v.to_device(&Device::Cpu)?);
        }
    }

    candle_core::safetensors::save(&tensor_map, &safetensors_path)?;

    // Write metadata
    let now = chrono_now();
    let metadata = CacheMetadata {
        session_id: session_id.to_string(),
        model_name: model_name.to_string(),
        processed_tokens: processed_tokens.to_vec(),
        layer_count: snapshots.len(),
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

/// Load KV cache snapshots from disk.
pub fn load_from_disk(
    dir: &Path,
    session_id: &str,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Vec<LayerKvSnapshot>, CacheMetadata)>> {
    let safetensors_path = dir.join(format!("{session_id}.safetensors"));
    let metadata_path = dir.join(format!("{session_id}.json"));

    if !safetensors_path.exists() || !metadata_path.exists() {
        return Ok(None);
    }

    let start = std::time::Instant::now();

    // Read metadata
    let json = std::fs::read_to_string(&metadata_path)?;
    let mut metadata: CacheMetadata = serde_json::from_str(&json)?;

    // Update last_accessed
    metadata.last_accessed = chrono_now();
    let updated_json = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(&metadata_path, updated_json)?;

    // Read tensors
    let tensors = candle_core::safetensors::load(&safetensors_path, device)?;

    // Reconstruct snapshots
    let mut snapshots = Vec::with_capacity(metadata.layer_count);
    for layer_idx in 0..metadata.layer_count {
        let k_name = format!("layer_{layer_idx}_k");
        let v_name = format!("layer_{layer_idx}_v");

        let k_data = tensors
            .get(&k_name)
            .map(|t| t.to_dtype(dtype))
            .transpose()?;
        let v_data = tensors
            .get(&v_name)
            .map(|t| t.to_dtype(dtype))
            .transpose()?;

        // Determine seq_len from tensor shape (dim 2 for KV cache format)
        let current_seq_len = k_data.as_ref().and_then(|t| t.dim(2).ok()).unwrap_or(0);

        snapshots.push(LayerKvSnapshot {
            layer_idx,
            is_sliding: false, // The model handles sliding vs normal on restore
            k_data,
            v_data,
            offset: 0,
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

/// Delete a session's cache files from disk.
pub fn delete_cache(dir: &Path, session_id: &str) -> Result<()> {
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

/// Remove old caches that exceed max_entries or max_age.
/// Returns the number of caches deleted.
pub fn expire_old_caches(dir: &Path, max_entries: usize, max_age: Duration) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }

    let mut entries: Vec<(String, CacheMetadata)> = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(json) = std::fs::read_to_string(&path) {
                if let Ok(meta) = serde_json::from_str::<CacheMetadata>(&json) {
                    entries.push((meta.session_id.clone(), meta));
                }
            }
        }
    }

    // Sort by last_accessed ascending (oldest first)
    entries.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));

    let now = chrono_now();
    let mut deleted = 0;

    for (session_id, meta) in &entries {
        let is_expired = is_older_than(&meta.last_accessed, &now, max_age);
        let over_capacity = entries.len() - deleted > max_entries;

        if is_expired || over_capacity {
            if let Err(e) = delete_cache(dir, session_id) {
                tracing::warn!("Failed to delete cache for session {session_id}: {e}");
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

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

fn is_older_than(timestamp_str: &str, now_str: &str, max_age: Duration) -> bool {
    let Ok(ts) = timestamp_str.parse::<u64>() else {
        return true;
    };
    let Ok(now) = now_str.parse::<u64>() else {
        return false;
    };
    now.saturating_sub(ts) > max_age.as_secs()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn chrono_now_returns_numeric_string() {
        let now = chrono_now();
        assert!(now.parse::<u64>().is_ok());
    }

    #[test]
    fn is_older_than_works() {
        assert!(is_older_than("1000", "5000", Duration::from_secs(3600)));
        assert!(!is_older_than("4000", "5000", Duration::from_secs(3600)));
    }

    #[test]
    fn expire_on_nonexistent_dir_returns_zero() {
        let result =
            expire_old_caches(Path::new("/nonexistent/path"), 8, Duration::from_secs(3600));
        assert_eq!(result.unwrap(), 0);
    }
}
