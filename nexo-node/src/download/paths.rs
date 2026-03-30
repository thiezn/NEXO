use std::path::PathBuf;
use std::sync::LazyLock;

/// Returns `~/.nexo/`.
pub fn nexo_home_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
}

/// Default model storage directory: `~/.nexo/models/`.
/// Overridable via `NEXO_NODE_MODELS_DIR` environment variable.
pub fn default_models_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = if let Ok(override_dir) = std::env::var("NEXO_NODE_MODELS_DIR") {
            PathBuf::from(override_dir)
        } else {
            nexo_home_dir().join("models")
        };
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

/// Return the storage directory for a specific model.
pub fn model_storage_dir(model_name: &str) -> PathBuf {
    default_models_dir().join(model_name.replace(':', "-"))
}
