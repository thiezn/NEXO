use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

/// Returns the root directory for local model storage.
///
/// The directory is created on first access if it does not already exist.
#[must_use]
pub(crate) fn default_models_dir() -> PathBuf {
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

/// Returns the Hugging Face cache directory inside the local model root.
///
/// The directory is created on first access if it does not already exist.
#[must_use]
pub(crate) fn hf_cache_dir() -> PathBuf {
    static DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = default_models_dir().join(".hf-cache");
        let _ = std::fs::create_dir_all(&dir);
        dir
    });

    DIR.clone()
}

/// Returns whether a path is a safe relative storage path.
///
/// # Arguments
///
/// * `path` - The path to validate for safe relative storage usage.
#[must_use]
pub(crate) fn is_relative_storage_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

#[cfg(test)]
mod tests {
    use super::is_relative_storage_path;
    use std::path::Path;

    #[test]
    fn rejects_absolute_and_parent_paths() {
        assert!(!is_relative_storage_path(Path::new("/tmp/model")));
        assert!(!is_relative_storage_path(Path::new("../model")));
    }

    #[test]
    fn accepts_normal_relative_paths() {
        assert!(is_relative_storage_path(Path::new("model")));
        assert!(is_relative_storage_path(Path::new("voices/af_alloy.pt")));
    }
}
