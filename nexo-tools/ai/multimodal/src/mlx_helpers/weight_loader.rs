use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use mlx_rs::Array;

/// Load all safetensors shard files and merge into a single weight map.
pub fn load_safetensors_shards(paths: &[&Path]) -> anyhow::Result<HashMap<String, Array>> {
    let mut weights = HashMap::new();
    for path in paths {
        let data = std::fs::read(path)
            .with_context(|| format!("reading safetensors: {}", path.display()))?;
        let shard = mlx_rs::safetensors::load_bytes(&data)?;
        for (key, array) in shard {
            weights.insert(key, array);
        }
    }
    tracing::info!(tensors = weights.len(), "loaded all weight shards");
    Ok(weights)
}

/// Extract a weight from the map, returning an error if missing.
pub fn get_weight(weights: &HashMap<String, Array>, key: &str) -> anyhow::Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing weight: {key}"))
}

/// Extract a weight from the map and cast to target dtype.
pub fn get_weight_as(
    weights: &HashMap<String, Array>,
    key: &str,
    dtype: mlx_rs::Dtype,
) -> anyhow::Result<Array> {
    let w = get_weight(weights, key)?;
    Ok(w.as_dtype(dtype)?)
}
