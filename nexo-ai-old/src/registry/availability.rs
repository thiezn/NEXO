#[cfg(feature = "download")]
use std::path::{Path, PathBuf};

#[cfg(feature = "download")]
use super::manifest::{AiModelManifest, known_manifests};

#[cfg(feature = "download")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ManifestRejection {
    Empty,
    MissingFile { path: PathBuf },
    HashMismatch { filename: String },
    HashVerificationFailed { filename: String, error: String },
}

#[cfg(feature = "download")]
impl ManifestRejection {
    fn message(&self) -> String {
        match self {
            Self::Empty => "manifest defines no files".to_string(),
            Self::MissingFile { path } => format!("missing file {}", path.display()),
            Self::HashMismatch { filename } => format!("SHA-256 mismatch for {filename}"),
            Self::HashVerificationFailed { filename, error } => {
                format!("failed to verify SHA-256 for {filename}: {error}")
            }
        }
    }
}

/// Scan the models directory to find which registered models are fully downloaded
/// and have valid SHA-256 checksums (where checksums are declared in the manifest).
#[cfg(feature = "download")]
pub fn detect_available_models() -> Vec<String> {
    use crate::download::paths::default_models_dir;
    use rayon::prelude::*;

    let models_dir = default_models_dir();
    known_manifests()
        .par_iter()
        .filter_map(|manifest| match manifest_rejection(manifest, &models_dir) {
            None => Some(manifest.manifest.name.clone()),
            Some(rejection) => {
                log_manifest_rejection(manifest, &models_dir, &rejection);
                None
            }
        })
        .collect()
}

#[cfg(feature = "download")]
fn manifest_rejection(manifest: &AiModelManifest, models_dir: &Path) -> Option<ManifestRejection> {
    use crate::download::manifest::storage_path;

    if manifest.manifest.files.is_empty() {
        return Some(ManifestRejection::Empty);
    }

    for file in &manifest.manifest.files {
        let path = models_dir.join(storage_path(&manifest.manifest, file));
        if !path.exists() {
            return Some(ManifestRejection::MissingFile { path });
        }

        if let Some(expected_hash) = file.sha256 {
            match crate::download::verify_sha256(&path, expected_hash) {
                Ok(true) => {}
                Ok(false) => {
                    return Some(ManifestRejection::HashMismatch {
                        filename: file.hf_filename.clone(),
                    });
                }
                Err(error) => {
                    return Some(ManifestRejection::HashVerificationFailed {
                        filename: file.hf_filename.clone(),
                        error: error.to_string(),
                    });
                }
            }
        }
    }

    None
}

#[cfg(feature = "download")]
fn log_manifest_rejection(
    manifest: &AiModelManifest,
    models_dir: &Path,
    rejection: &ManifestRejection,
) {
    let reason = rejection.message();
    match rejection {
        ManifestRejection::Empty => {
            tracing::debug!(
                "Rejecting model manifest '{}' during disk detection: {reason}",
                manifest.manifest.name,
            );
        }
        ManifestRejection::MissingFile { .. } if manifest_has_local_artifacts(manifest, models_dir) => {
            tracing::info!(
                "Rejecting model manifest '{}' during disk detection: {reason}",
                manifest.manifest.name,
            );
        }
        ManifestRejection::MissingFile { .. } => {
            tracing::debug!(
                "Rejecting model manifest '{}' during disk detection: {reason}",
                manifest.manifest.name,
            );
        }
        ManifestRejection::HashMismatch { .. }
        | ManifestRejection::HashVerificationFailed { .. } => {
            tracing::warn!(
                "Rejecting model manifest '{}' during disk detection: {reason}",
                manifest.manifest.name,
            );
        }
    }
}

#[cfg(feature = "download")]
fn manifest_has_local_artifacts(manifest: &AiModelManifest, models_dir: &Path) -> bool {
    use crate::download::manifest::storage_path;

    manifest.manifest.files.iter().any(|file| {
        let path = models_dir.join(storage_path(&manifest.manifest, file));
        path.exists() || path.parent().is_some_and(Path::exists)
    })
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::api::types::ModelCategory;
    use crate::download::{ModelFile, ModelManifest};
    use crate::registry::AiComponent;
    use crate::registry::manifest::{ModelFamily, ModelRuntime, OpenAiProvider};

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
                "nexo-ai-registry-scan-{label}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn test_manifest(files: Vec<ModelFile<AiComponent>>) -> AiModelManifest {
        AiModelManifest {
            manifest: ModelManifest {
                name: "test-model".to_string(),
                family: ModelFamily::Gemma4.as_str().to_string(),
                description: "test".to_string(),
                size_gb: 1.0,
                files,
            },
            family: ModelFamily::Gemma4,
            runtime: ModelRuntime::OpenAi {
                provider: OpenAiProvider::MlxVlm,
                model_repo: "mlx-community/test-model".to_string(),
            },
            categories: vec![ModelCategory::Chat],
        }
    }

    #[test]
    fn manifest_rejection_reports_missing_file_when_partial_model_present() {
        let temp = TempDir::new("missing-file");
        let manifest = test_manifest(vec![ModelFile {
            component: AiComponent::Config,
            hf_repo: "mlx-community/test-model".to_string(),
            hf_filename: "config.json".to_string(),
            size_bytes: 1,
            gated: false,
            sha256: None,
        }]);

        std::fs::create_dir_all(temp.path.join("test-model")).unwrap();

        let rejection = manifest_rejection(&manifest, &temp.path);

        assert!(matches!(
            rejection,
            Some(ManifestRejection::MissingFile { .. })
        ));
        assert!(manifest_has_local_artifacts(&manifest, &temp.path));
    }

    #[test]
    fn manifest_rejection_accepts_complete_manifest() {
        let temp = TempDir::new("complete-manifest");
        let manifest = test_manifest(vec![ModelFile {
            component: AiComponent::Config,
            hf_repo: "mlx-community/test-model".to_string(),
            hf_filename: "config.json".to_string(),
            size_bytes: 1,
            gated: false,
            sha256: None,
        }]);

        let config_path = temp.path.join("test-model").join("config.json");
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(&config_path, []).unwrap();

        assert_eq!(manifest_rejection(&manifest, &temp.path), None);
    }
}
