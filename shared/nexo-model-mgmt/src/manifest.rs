//! Manifest data structures and local storage path mapping.

use std::path::{Path, PathBuf};

use nexo_core::ModelDescriptor;
use serde_json::Value;

/// Component types for files that make up a model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    /// Primary model weight file.
    Weights,
    /// Sharded model weight file.
    WeightShard,
    /// Mistral UQFF quantized artifact.
    UqffShard,
    /// Non-quantized residual tensors needed next to UQFF artifacts.
    UqffResidual,
    /// Tokenizer file.
    Tokenizer,
    /// Tokenizer sidecar configuration.
    TokenizerConfig,
    /// Model configuration or auxiliary metadata file.
    Config,
    /// Generation defaults.
    GenerationConfig,
    /// Chat template file.
    ChatTemplate,
    /// Multimodal processor configuration.
    ProcessorConfig,
    /// Multimodal preprocessor configuration.
    PreprocessorConfig,
    /// Embedding module manifest.
    Modules,
    /// Multimodal projector weights.
    VisionProjector,
}

impl ModelComponent {
    /// Returns the stable component identifier used in CLI output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::WeightShard => "weight_shard",
            Self::UqffShard => "uqff_shard",
            Self::UqffResidual => "uqff_residual",
            Self::Tokenizer => "tokenizer",
            Self::TokenizerConfig => "tokenizer_config",
            Self::Config => "config",
            Self::GenerationConfig => "generation_config",
            Self::ChatTemplate => "chat_template",
            Self::ProcessorConfig => "processor_config",
            Self::PreprocessorConfig => "preprocessor_config",
            Self::Modules => "modules",
            Self::VisionProjector => "vision_projector",
        }
    }
}

/// How a manifest selects files from a remote model repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFileSelector {
    /// A single known path in the remote repository.
    Exact(String),
    /// Every remote path ending with this suffix.
    Suffix(String),
    /// Every remote path starting with this prefix.
    Prefix(String),
}

impl ModelFileSelector {
    /// Returns whether a remote filename is matched by this selector.
    #[must_use]
    pub fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(exact) => filename == exact,
            Self::Suffix(suffix) => filename.ends_with(suffix),
            Self::Prefix(prefix) => filename.starts_with(prefix),
        }
    }

    /// Returns the exact remote path when this selector is exact.
    #[must_use]
    pub fn exact_path(&self) -> Option<&str> {
        match self {
            Self::Exact(path) => Some(path),
            Self::Suffix(_) | Self::Prefix(_) => None,
        }
    }

    /// Human-readable selector label used in diagnostics.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Exact(path) | Self::Suffix(path) | Self::Prefix(path) => path,
        }
    }
}

/// A single file to download from Hugging Face.
#[derive(Debug, Clone)]
pub struct ModelFile {
    /// The logical component this file belongs to.
    pub component: ModelComponent,
    /// Hugging Face model repository.
    pub hf_repo: String,
    /// Path selector inside the Hugging Face repository.
    pub selector: ModelFileSelector,
    /// Optional clean path under this manifest's local storage directory.
    pub local_path: Option<String>,
    /// Expected file size in bytes, when known.
    pub size_bytes: Option<u64>,
    /// Whether the source repository is gated.
    pub gated: bool,
    /// Expected SHA-256 hex digest. `None` means no digest is pinned yet.
    pub sha256: Option<&'static str>,
}

/// A complete model definition: canonical descriptor, runtime metadata, and files.
#[derive(Debug, Clone)]
pub struct ModelManifest {
    /// Canonical model identity and capabilities shared with the rest of Nexo.
    pub descriptor: ModelDescriptor,
    /// Loader/runtime backend label.
    pub backend: String,
    /// Approximate total download size in gigabytes.
    pub size_gb: f32,
    /// Files required for this model.
    pub files: Vec<ModelFile>,
}

impl ModelManifest {
    /// Stable local model id used by `models pull` and storage paths.
    #[must_use]
    pub fn id(&self) -> &str {
        self.descriptor.id.as_str()
    }

    /// Human-readable display name.
    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.descriptor.display_name
    }

    /// Shared local storage key used for downloaded artifacts.
    #[must_use]
    pub fn storage_id(&self) -> String {
        self.descriptor
            .metadata
            .get("source_model")
            .and_then(Value::as_str)
            .map(base_storage_id)
            .unwrap_or_else(|| sanitize_model_name(self.id()))
    }
}

/// Determine the clean storage path for a remote filename relative to the models directory.
#[must_use]
pub fn storage_path(manifest: &ModelManifest, remote_filename: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(manifest.storage_id()).join(remote_filename)
}

/// Determine the clean storage path for a remote file, honoring per-file local path overrides.
#[must_use]
pub fn storage_path_for_file(
    manifest: &ModelManifest,
    file: &ModelFile,
    remote_filename: impl AsRef<Path>,
) -> PathBuf {
    let relative_path = file
        .local_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| remote_filename.as_ref().to_path_buf());
    PathBuf::from(manifest.storage_id()).join(relative_path)
}

fn base_storage_id(source_model: &str) -> String {
    let basename = source_model
        .rsplit_once('/')
        .map_or(source_model, |(_, name)| name);
    sanitize_model_name(&basename.to_ascii_lowercase())
}

/// Sanitize a model name for local directory storage.
#[must_use]
pub fn sanitize_model_name(model_name: &str) -> String {
    model_name.replace([':', '/'], "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_path_preserves_subdirectories() {
        let manifest = ModelManifest {
            descriptor: nexo_core::ModelDescriptor {
                id: "org/test:model".into(),
                display_name: "Test model".to_string(),
                provider: Some("test".to_string()),
                capabilities: vec![nexo_core::ModelCapability::TextGeneration],
                modalities: nexo_core::ModelModalities {
                    input: vec![nexo_core::SupportedModality::Text],
                    output: vec![nexo_core::SupportedModality::Text],
                },
                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: Default::default(),
            },
            backend: "candle-gguf".to_string(),
            size_gb: 1.0,
            files: Vec::new(),
        };
        let file = ModelFile {
            component: ModelComponent::Weights,
            hf_repo: "org/test".to_string(),
            selector: ModelFileSelector::Exact("subdir/model.gguf".to_string()),
            local_path: None,
            size_bytes: Some(1),
            gated: false,
            sha256: None,
        };

        assert_eq!(
            storage_path(&manifest, file.selector.exact_path().unwrap_or_default()),
            PathBuf::from("org-test-model/subdir/model.gguf")
        );
    }

    #[test]
    fn storage_id_uses_source_model_basename() {
        let mut metadata = nexo_core::MetadataMap::new();
        metadata.insert(
            "source_model".to_string(),
            Value::String("google/gemma-4-E4B-it".to_string()),
        );

        let manifest = ModelManifest {
            descriptor: nexo_core::ModelDescriptor {
                id: "gemma-4-e4b-it-uqff-afq8".into(),
                display_name: "Gemma 4 E4B IT UQFF AFQ8".to_string(),
                provider: Some("mistralrs-community".to_string()),
                capabilities: vec![nexo_core::ModelCapability::TextGeneration],
                modalities: nexo_core::ModelModalities {
                    input: vec![nexo_core::SupportedModality::Text],
                    output: vec![nexo_core::SupportedModality::Text],
                },
                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata,
            },
            backend: "mistralrs-uqff".to_string(),
            size_gb: 1.0,
            files: Vec::new(),
        };

        assert_eq!(manifest.storage_id(), "gemma-4-e4b-it");
    }

    #[test]
    fn storage_path_uses_file_local_path_override() {
        let manifest = ModelManifest {
            descriptor: nexo_core::ModelDescriptor {
                id: "dia-1.6b-tts".into(),
                display_name: "Dia 1.6B TTS".to_string(),
                provider: Some("nari-labs".to_string()),
                capabilities: vec![nexo_core::ModelCapability::SpeechGeneration],
                modalities: nexo_core::ModelModalities {
                    input: vec![nexo_core::SupportedModality::Text],
                    output: vec![nexo_core::SupportedModality::Audio],
                },
                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: Default::default(),
            },
            backend: "mistralrs-dia".to_string(),
            size_gb: 1.0,
            files: Vec::new(),
        };
        let file = ModelFile {
            component: ModelComponent::Weights,
            hf_repo: "EricB/dac_44khz".to_string(),
            selector: ModelFileSelector::Exact("model.safetensors".to_string()),
            local_path: Some("dac/model.safetensors".to_string()),
            size_bytes: None,
            gated: false,
            sha256: None,
        };

        assert_eq!(
            storage_path_for_file(&manifest, &file, "model.safetensors"),
            PathBuf::from("dia-1.6b-tts/dac/model.safetensors")
        );
    }
}
