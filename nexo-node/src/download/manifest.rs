use std::path::PathBuf;

/// Trait for model component identifiers.
pub trait Component: Clone + std::fmt::Debug + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn is_model_specific(&self) -> bool;
}

/// Component types for GGUF node model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufComponent {
    /// The GGUF weights file.
    Weights,
    /// The multimodal vision projector (mmproj) file for image analysis.
    VisionProjector,
}

impl Component for GgufComponent {
    fn name(&self) -> &str {
        match self {
            Self::Weights => "weights",
            Self::VisionProjector => "vision_projector",
        }
    }

    fn is_model_specific(&self) -> bool {
        true
    }
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
/// Model names are sanitized: colons become dashes.
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_manifest() -> ModelManifest<GgufComponent> {
        ModelManifest {
            name: "qwen3.5-35b-ab3b".to_string(),
            family: "qwen3.5".to_string(),
            description: "Test".to_string(),
            size_gb: 20.0,
            files: vec![],
        }
    }

    #[test]
    fn model_specific_storage_path() {
        let manifest = test_manifest();
        let file = ModelFile {
            component: GgufComponent::Weights,
            hf_repo: "repo".to_string(),
            hf_filename: "model.gguf".to_string(),
            size_bytes: 100,
            gated: false,
            sha256: None,
        };
        let path = storage_path(&manifest, &file);
        assert_eq!(path, PathBuf::from("qwen3.5-35b-ab3b/model.gguf"));
    }
}
