use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;

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

/// Find a single `.gguf` file in a directory matching a pattern.
///
/// The `pattern` is matched against filenames using simple `contains` logic.
/// For example, `"Q5_K_M"` matches `"Qwen3-4B-Q5_K_M.gguf"`.
///
/// Files whose name starts with any entry in `exclude_prefixes` are skipped.
pub fn find_gguf_file(model_dir: &Path, pattern: &str, exclude_prefixes: &[&str]) -> Result<PathBuf> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "gguf")
                && p.file_name()
                    .and_then(|f| f.to_str())
                    .is_some_and(|name| {
                        name.contains(pattern)
                            && !exclude_prefixes.iter().any(|pfx| name.starts_with(pfx))
                    })
        })
        .collect();

    match matches.len() {
        0 => anyhow::bail!(
            "no .gguf file matching '{}' in {}",
            pattern,
            model_dir.display()
        ),
        1 => Ok(matches.remove(0)),
        n => anyhow::bail!(
            "expected 1 .gguf file matching '{}' in {}, found {}",
            pattern,
            model_dir.display(),
            n
        ),
    }
}

/// Open and parse a GGUF file, returning the content metadata and an open file handle.
pub fn load_gguf(path: &Path) -> Result<(gguf_file::Content, File)> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open GGUF file: {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to parse GGUF file {}: {e}", path.display()))?;
    Ok((content, file))
}
