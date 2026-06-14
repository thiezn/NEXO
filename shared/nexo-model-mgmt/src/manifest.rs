//! Manifest data structures and local storage path mapping.

use std::path::{Path, PathBuf};

use nexo_core::{InferenceRuntime, ModelDefinition};
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
    ///
    /// This happens when the model publisher restricts
    /// access to the repository, for example by requiring
    /// users to accept a license on Hugging Face before downloading.
    pub gated: bool,

    /// Expected SHA-256 hex digest. `None` means no digest is pinned yet.
    pub sha256: Option<&'static str>,
}

/// Data type preference declared by a model manifest for runtime loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ManifestModelDataType {
    /// Let the runtime choose the best supported type.
    #[default]
    Auto,
    /// Prefer BF16 weights or activations.
    Bf16,
    /// Prefer F16 weights or activations.
    F16,
    /// Prefer F32 weights or activations.
    F32,
}

/// Runtime binding declared by a model manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestRuntimeBinding {
    /// Bind the model to the internal AnyTTS adapter runtime.
    AnyTts(AnyTtsManifestBinding),
    /// Bind the model to Mistral.rs.
    MistralRs(MistralRsManifestBinding),
    /// Bind the model to mold-ai-inference.
    Mold(MoldManifestBinding),
}

impl ManifestRuntimeBinding {
    /// Returns the NEXO runtime used by this binding.
    #[must_use]
    pub const fn runtime(&self) -> InferenceRuntime {
        match self {
            Self::AnyTts(_) => InferenceRuntime::AnyTts,
            Self::MistralRs(_) => InferenceRuntime::MistralRs,
            Self::Mold(_) => InferenceRuntime::Mold,
        }
    }

    /// Returns a compact label used in CLI output.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::AnyTts(binding) => binding.label(),
            Self::MistralRs(binding) => binding.label(),
            Self::Mold(binding) => binding.label(),
        }
    }
}

/// Manifest binding data for AnyTTS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnyTtsManifestBinding {
    /// The internal AnyTTS engine implementation.
    pub engine: AnyTtsManifestEngine,
}

impl AnyTtsManifestBinding {
    const fn label(&self) -> &'static str {
        match self.engine {
            AnyTtsManifestEngine::Kokoro => "any_tts/kokoro",
        }
    }
}

/// Internal AnyTTS engines supported by manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnyTtsManifestEngine {
    /// Kokoro TTS.
    Kokoro,
}

/// Manifest binding data for Mistral.rs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralRsManifestBinding {
    /// Loader intent for the Mistral.rs runtime.
    pub loader: MistralRsManifestLoader,
    /// Optional Hugging Face revision to pin.
    pub revision: Option<String>,
}

impl MistralRsManifestBinding {
    const fn label(&self) -> &'static str {
        match self.loader {
            MistralRsManifestLoader::Auto(_) => "mistral_rs/auto",
            MistralRsManifestLoader::Gguf(_) => "mistral_rs/gguf",
            MistralRsManifestLoader::Diffusion(_) => "mistral_rs/diffusion",
            MistralRsManifestLoader::Speech(_) => "mistral_rs/speech",
        }
    }
}

/// Mistral.rs loader intent declared by manifests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MistralRsManifestLoader {
    /// Let Mistral.rs auto-detect a local model layout.
    Auto(MistralRsAutoManifestLoader),
    /// Load explicit GGUF files.
    Gguf(MistralRsGgufManifestLoader),
    /// Load a diffusion image-generation model.
    Diffusion(MistralRsDiffusionManifestLoader),
    /// Load a speech-generation model.
    Speech(MistralRsSpeechManifestLoader),
}

/// Manifest settings for Mistral.rs automatic loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralRsAutoManifestLoader {
    /// Optional UQFF shard filename prefixes or paths.
    pub from_uqff: Option<Vec<String>>,
    /// Preferred runtime dtype.
    pub dtype: ManifestModelDataType,
}

/// Manifest settings for Mistral.rs GGUF loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralRsGgufManifestLoader {
    /// Exact GGUF filenames to load from local storage.
    pub quantized_filenames: Vec<String>,
    /// Preferred activation dtype.
    pub dtype: ManifestModelDataType,
}

/// Manifest settings for Mistral.rs diffusion loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralRsDiffusionManifestLoader {
    /// Prefer the offloaded loader variant.
    pub offload: bool,
    /// Preferred runtime dtype.
    pub dtype: ManifestModelDataType,
}

/// Manifest settings for Mistral.rs speech loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralRsSpeechManifestLoader {
    /// Optional local DAC artifact subdirectory.
    pub dac_subdir: Option<String>,
    /// Preferred runtime dtype.
    pub dtype: ManifestModelDataType,
}

/// Manifest binding data for mold-ai-inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoldManifestBinding {
    /// Loader intent for mold.
    pub loader: MoldManifestLoader,
}

impl MoldManifestBinding {
    const fn label(&self) -> &'static str {
        match self.loader {
            MoldManifestLoader::Flux2 => "mold/flux2",
        }
    }
}

/// mold loader intent declared by manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoldManifestLoader {
    /// Load a FLUX.2 image-generation model.
    Flux2,
}

// /// A complete model definition: canonical descriptor, runtime metadata, and files.
// #[derive(Debug, Clone)]
// pub struct ModelManifest {
//     /// Canonical model identity and capabilities shared with the rest of Nexo.
//     pub descriptor: ModelDefinition,
//     /// Runtime bindings supported by this manifest.
//     pub runtime_bindings: Vec<ManifestRuntimeBinding>,
//     /// Approximate total download size in gigabytes.
//     pub size_gb: f32,
//     /// Files required for this model.
//     pub files: Vec<ModelFile>,
// }

// impl ModelManifest {
//     /// Stable local model id used by `models pull` and storage paths.
//     #[must_use]
//     pub fn id(&self) -> &str {
//         self.descriptor.id.as_str()
//     }

//     /// Human-readable display name.
//     #[must_use]
//     pub fn display_name(&self) -> &str {
//         &self.descriptor.display_name
//     }

//     /// Shared local storage key used for downloaded artifacts.
//     #[must_use]
//     pub fn storage_id(&self) -> String {
//         self.descriptor
//             .metadata
//             .get("source_model")
//             .and_then(Value::as_str)
//             .map(base_storage_id)
//             .unwrap_or_else(|| sanitize_model_name(self.id()))
//     }
// }

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
            descriptor: nexo_core::ModelDefinition {
                id: "org/test:model".into(),
                display_name: "Test model".to_string(),
                provider: Some("test".to_string()),
                runtime: nexo_core::InferenceRuntime::MistralRs,
                capabilities: vec![nexo_core::ModelCapability::TextGeneration],

                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: Default::default(),
            },
            runtime_bindings: vec![ManifestRuntimeBinding::MistralRs(
                MistralRsManifestBinding {
                    loader: MistralRsManifestLoader::Gguf(MistralRsGgufManifestLoader {
                        quantized_filenames: vec!["subdir/model.gguf".to_string()],
                        dtype: ManifestModelDataType::Auto,
                    }),
                    revision: None,
                },
            )],
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
            descriptor: nexo_core::ModelDefinition {
                id: "gemma-4-e4b-it-uqff-afq8".into(),
                display_name: "Gemma 4 E4B IT UQFF AFQ8".to_string(),
                provider: Some("mistralrs-community".to_string()),
                runtime: nexo_core::InferenceRuntime::MistralRs,
                capabilities: vec![nexo_core::ModelCapability::TextGeneration],

                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata,
            },
            runtime_bindings: vec![ManifestRuntimeBinding::MistralRs(
                MistralRsManifestBinding {
                    loader: MistralRsManifestLoader::Auto(MistralRsAutoManifestLoader {
                        from_uqff: Some(vec!["afq8-".to_string()]),
                        dtype: ManifestModelDataType::Auto,
                    }),
                    revision: None,
                },
            )],
            size_gb: 1.0,
            files: Vec::new(),
        };

        assert_eq!(manifest.storage_id(), "gemma-4-e4b-it");
    }

    #[test]
    fn storage_path_uses_file_local_path_override() {
        let manifest = ModelManifest {
            descriptor: nexo_core::ModelDefinition {
                id: "dia-1.6b-tts".into(),
                display_name: "Dia 1.6B TTS".to_string(),
                provider: Some("nari-labs".to_string()),
                runtime: nexo_core::InferenceRuntime::MistralRs,
                capabilities: vec![nexo_core::ModelCapability::SpeechGeneration],
                role_strategy: nexo_core::RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: Default::default(),
            },
            runtime_bindings: vec![ManifestRuntimeBinding::MistralRs(
                MistralRsManifestBinding {
                    loader: MistralRsManifestLoader::Speech(MistralRsSpeechManifestLoader {
                        dac_subdir: Some("dac".to_string()),
                        dtype: ManifestModelDataType::F16,
                    }),
                    revision: None,
                },
            )],
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
