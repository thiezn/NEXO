use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;

/// Find all `.safetensors` model weight files in a directory.
///
/// If `model.safetensors` exists, returns just that file.
/// Otherwise collects all `model-*.safetensors` shards, sorted by name.
/// Some image models store a single `diffusion_pytorch_model.safetensors`
/// file in a component subdirectory, which is also accepted.
pub fn find_safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut all_safetensors: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    all_safetensors.sort();

    let mut shards: Vec<PathBuf> = all_safetensors
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        })
        .cloned()
        .collect();

    if !shards.is_empty() {
        shards.sort();
        return Ok(shards);
    }

    let diffusion = model_dir.join("diffusion_pytorch_model.safetensors");
    if diffusion.exists() {
        return Ok(vec![diffusion]);
    }

    if all_safetensors.len() == 1 {
        return Ok(vec![all_safetensors.remove(0)]);
    }

    if all_safetensors.is_empty() {
        anyhow::bail!("no safetensor files found in {}", model_dir.display());
    }

    let names = all_safetensors
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::bail!(
        "found unexpected safetensor layout in {}: {}",
        model_dir.display(),
        names
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "nexo-ai-weights-{label}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn touch(&self, name: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, []).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn finds_diffusion_pytorch_model_file() {
        let dir = TempDir::new("diffusion");
        let expected = dir.touch("diffusion_pytorch_model.safetensors");

        let paths = find_safetensor_files(&dir.path).unwrap();

        assert_eq!(paths, vec![expected]);
    }

    #[test]
    fn finds_model_shards_in_order() {
        let dir = TempDir::new("shards");
        let second = dir.touch("model-00002-of-00002.safetensors");
        let first = dir.touch("model-00001-of-00002.safetensors");

        let paths = find_safetensor_files(&dir.path).unwrap();

        assert_eq!(paths, vec![first, second]);
    }

    #[test]
    fn accepts_single_nonstandard_safetensor_file() {
        let dir = TempDir::new("single");
        let expected = dir.touch("weights.safetensors");

        let paths = find_safetensor_files(&dir.path).unwrap();

        assert_eq!(paths, vec![expected]);
    }
}

/// Find a single `.gguf` file in a directory matching a pattern.
///
/// The `pattern` is matched against filenames using simple `contains` logic.
/// For example, `"Q5_K_M"` matches `"Qwen3-4B-Q5_K_M.gguf"`.
///
/// Files whose name starts with any entry in `exclude_prefixes` are skipped.
pub fn find_gguf_file(
    model_dir: &Path,
    pattern: &str,
    exclude_prefixes: &[&str],
) -> Result<PathBuf> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "gguf")
                && p.file_name().and_then(|f| f.to_str()).is_some_and(|name| {
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
