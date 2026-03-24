use std::path::PathBuf;
use std::sync::LazyLock;

/// Default model storage directory: `~/.nexo/local_models/`.
pub fn default_models_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = if let Ok(override_dir) = std::env::var("NEXO_AI_MODELS_DIR") {
            PathBuf::from(override_dir)
        } else {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".nexo")
                .join("local_models")
        };
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

/// Internal hf-hub cache directory: `<models_dir>/.hf-cache/`.
/// Hidden from users; files get hardlinked to clean paths after download.
#[cfg(feature = "download")]
pub(crate) fn hf_cache_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = default_models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_models_dir_is_under_nexo() {
        let dir = default_models_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains(".nexo") && dir_str.contains("local_models"),
            "expected .nexo/local_models in path, got: {dir_str}"
        );
    }
}
