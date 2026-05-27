use std::path::Path;

use super::manifest::{AiModelManifest, known_manifests};
use crate::api::types::ModelCategory;
use crate::download::manifest::storage_path;
use crate::download::paths::default_models_dir;

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub family: String,
    pub backend: String,
    pub categories: Vec<ModelCategory>,
    pub size_gb: f32,
    pub description: String,
    pub is_downloaded: bool,
    pub is_loaded: bool,
}

pub(crate) fn manifest_is_downloaded(manifest: &AiModelManifest, models_dir: &Path) -> bool {
    !manifest.manifest.files.is_empty()
        && manifest.manifest.files.iter().all(|file| {
            models_dir
                .join(storage_path(&manifest.manifest, file))
                .exists()
        })
}

/// Build a list of all known models, checking download status and
/// using the provided closure to determine if each model is currently loaded.
pub fn list_models(is_loaded: impl Fn(&str) -> bool) -> Vec<ModelEntry> {
    let models_dir = default_models_dir();

    known_manifests()
        .iter()
        .map(|m| {
            let name = &m.manifest.name;
            ModelEntry {
                name: name.clone(),
                family: m.family.to_string(),
                backend: m.runtime.as_str().to_string(),
                categories: m.categories.clone(),
                size_gb: m.manifest.size_gb,
                description: m.manifest.description.clone(),
                is_downloaded: manifest_is_downloaded(m, &models_dir),
                is_loaded: is_loaded(name),
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::download::{ModelFile, ModelManifest};
    use crate::registry::manifest::{AiComponent, ModelFamily, ModelRuntime, OpenAiProvider};

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
                "nexo-ai-registry-{label}-{}-{unique}",
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
                name: "test-openai-model".to_string(),
                family: ModelFamily::Whisper.as_str().to_string(),
                description: "test".to_string(),
                size_gb: 1.0,
                files,
            },
            family: ModelFamily::Whisper,
            runtime: ModelRuntime::OpenAi {
                provider: OpenAiProvider::MlxAudio,
                model_repo: "mlx-community/test".to_string(),
            },
            categories: vec![ModelCategory::Listen],
        }
    }

    #[test]
    fn openai_models_without_files_are_not_downloaded() {
        let temp = TempDir::new("empty-openai");
        let manifest = test_manifest(Vec::new());

        assert!(!manifest_is_downloaded(&manifest, &temp.path));
    }

    #[test]
    fn model_is_downloaded_only_when_all_manifest_files_exist() {
        let temp = TempDir::new("complete-openai");
        let manifest = test_manifest(vec![
            ModelFile {
                component: AiComponent::Config,
                hf_repo: "mlx-community/test".to_string(),
                hf_filename: "config.json".to_string(),
                size_bytes: 1,
                gated: false,
                sha256: None,
            },
            ModelFile {
                component: AiComponent::Model,
                hf_repo: "mlx-community/test".to_string(),
                hf_filename: "model.safetensors".to_string(),
                size_bytes: 1,
                gated: false,
                sha256: None,
            },
        ]);

        let config_path = temp.path.join("test-openai-model").join("config.json");
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(&config_path, []).unwrap();
        assert!(!manifest_is_downloaded(&manifest, &temp.path));

        let model_path = temp
            .path
            .join("test-openai-model")
            .join("model.safetensors");
        std::fs::write(&model_path, []).unwrap();
        assert!(manifest_is_downloaded(&manifest, &temp.path));
    }
}
