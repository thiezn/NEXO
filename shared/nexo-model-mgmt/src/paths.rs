//! Local filesystem paths for downloaded model artifacts.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::manifest::sanitize_model_name;
use crate::registry::find_manifest_by_source;

/// Default model storage directory: `~/.nexo/local_models/`.
#[must_use]
pub fn default_models_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = std::env::var("NEXO_AI_MODELS_DIR").map_or_else(
            |_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".nexo")
                    .join("local_models")
            },
            PathBuf::from,
        );
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}

/// Return the storage directory for a specific local model name.
#[must_use]
pub fn model_storage_dir(model_name: &str) -> PathBuf {
    default_models_dir().join(sanitize_model_name(model_name))
}

/// Resolve either an existing filesystem path, a manifest name, or a manifest source repo to a local directory.
#[must_use]
pub fn resolve_model_storage_dir(model_or_path: &str) -> PathBuf {
    let path = Path::new(model_or_path);
    if path.exists() {
        return path.to_path_buf();
    }

    find_manifest_by_source(model_or_path)
        .map(|manifest| default_models_dir().join(manifest.storage_id()))
        .unwrap_or_else(|| model_storage_dir(model_or_path))
}

pub(crate) fn hf_cache_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = default_models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    });
    DIR.clone()
}
