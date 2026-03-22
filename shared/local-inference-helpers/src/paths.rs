use std::path::PathBuf;
use std::sync::OnceLock;

/// Default model storage directory: `~/.myclaw/local_models/`.
pub fn default_models_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = if let Ok(override_dir) = std::env::var("LOCAL_INFERENCE_MODELS_DIR") {
            PathBuf::from(override_dir)
        } else {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".myclaw")
                .join("local_models")
        };
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
    .clone()
}

/// Internal hf-hub cache directory: `<models_dir>/.hf-cache/`.
/// Hidden from users; files get hardlinked to clean paths after download.
pub fn hf_cache_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = default_models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
    .clone()
}
