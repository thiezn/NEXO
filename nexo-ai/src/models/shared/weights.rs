use std::path::{Path, PathBuf};

use anyhow::Result;

/// Find all `.safetensors` model weight files in a directory.
///
/// If `model.safetensors` exists, returns just that file.
/// Otherwise collects all `model-*.safetensors` shards, sorted by name.
pub fn find_safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        })
        .collect();

    if shards.is_empty() {
        anyhow::bail!(
            "no safetensor files found in {}",
            model_dir.display()
        );
    }

    shards.sort();
    Ok(shards)
}
