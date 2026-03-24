use std::path::PathBuf;

/// Trait for model component identifiers. Each consumer provides its own enum.
pub trait Component: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Short identifier used as storage key (e.g. "model", "tokenizer").
    fn name(&self) -> &str;

    /// Whether this component is model-specific (stored per-model) or shared
    /// across models of the same family (stored under `shared/<family>/`).
    fn is_model_specific(&self) -> bool;
}

/// A single file to download from HuggingFace.
#[derive(Debug, Clone)]
pub struct ModelFile<C: Component> {
    pub component: C,
    pub hf_repo: String,
    pub hf_filename: String,
    pub size_bytes: u64,
    pub gated: bool,
    /// Expected SHA-256 hex digest. None means not yet collected.
    pub sha256: Option<&'static str>,
}

/// A complete model definition: identity + files to download.
#[derive(Debug, Clone)]
pub struct ModelManifest<C: Component> {
    pub name: String,
    pub family: String,
    pub description: String,
    pub size_gb: f32,
    pub files: Vec<ModelFile<C>>,
}

/// Determine the clean storage path for a model file relative to the models directory.
///
/// - Model-specific components: `<model-name>/<hf_filename>`
/// - Shared components: `shared/<family>/<hf_filename>`
///
/// Model names are sanitized: colons become dashes (e.g. `flux-schnell:q8` -> `flux-schnell-q8`).
pub fn storage_path<C: Component>(manifest: &ModelManifest<C>, file: &ModelFile<C>) -> PathBuf {
    let sanitized_name = manifest.name.replace(':', "-");

    if file.component.is_model_specific() {
        PathBuf::from(&sanitized_name).join(&file.hf_filename)
    } else {
        PathBuf::from("shared")
            .join(&manifest.family)
            .join(&file.hf_filename)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    enum TestComponent {
        Model,
        Shared,
    }

    impl Component for TestComponent {
        fn name(&self) -> &str {
            match self {
                Self::Model => "model",
                Self::Shared => "shared",
            }
        }

        fn is_model_specific(&self) -> bool {
            matches!(self, Self::Model)
        }
    }

    fn test_manifest() -> ModelManifest<TestComponent> {
        ModelManifest {
            name: "test-model:q8".to_string(),
            family: "test".to_string(),
            description: "Test model".to_string(),
            size_gb: 1.0,
            files: vec![],
        }
    }

    #[test]
    fn model_specific_storage_path() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Model,
            hf_repo: "repo".to_string(),
            hf_filename: "weights.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(path, PathBuf::from("test-model-q8/weights.safetensors"));
    }

    #[test]
    fn shared_storage_path() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Shared,
            hf_repo: "repo".to_string(),
            hf_filename: "vae.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(path, PathBuf::from("shared/test/vae.safetensors"));
    }

    #[test]
    fn storage_path_preserves_subdirectory() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: TestComponent::Model,
            hf_repo: "repo".to_string(),
            hf_filename: "subfolder/model.safetensors".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(
            path,
            PathBuf::from("test-model-q8/subfolder/model.safetensors")
        );
    }
}
