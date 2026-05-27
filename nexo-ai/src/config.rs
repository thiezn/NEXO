use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, ModelDescriptor, Result};

/// Serializable crate configuration for a library-first `nexo-ai` runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NexoAiConfig {
    /// Runtime-wide settings shared by all configured models.
    pub runtime: RuntimeConfig,

    /// The models exposed through the local registry and runtime.
    pub models: Vec<RegisteredModelConfig>,
}

impl NexoAiConfig {
    /// Loads the runtime configuration from the given path, creating a default file when absent.
    ///
    /// # Arguments
    ///
    /// * `path` - The configuration file path to load.
    pub fn load(path: &Path) -> Result<Self> {
        cli_helpers::config::load_or_create(path).map_err(|error| Error::Config {
            message: error.to_string(),
        })
    }

    /// Saves the runtime configuration to the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The configuration file path to write.
    pub fn save(&self, path: &Path) -> Result {
        cli_helpers::config::save(self, path).map_err(|error| Error::Config {
            message: error.to_string(),
        })
    }
}

/// Runtime-wide settings that govern model loading and request scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// The target device selection policy used for model loading.
    pub device: DeviceSpec,

    /// The scheduler policy used for generation requests.
    pub scheduler: SchedulerPolicy,

    /// Disables KV-cache use in the underlying runtime when set.
    pub no_kv_cache: bool,

    /// Disables prefix caching in the underlying runtime when set.
    pub no_prefix_cache: bool,

    /// The number of prefix-cache entries to retain when prefix caching is enabled.
    pub prefix_cache_entries: usize,

    /// Disables EOS-based stopping if the selected runtime should continue beyond EOS.
    pub disable_eos_stop: bool,

    /// Enables periodic throughput logging from `mistralrs-core`.
    pub throughput_logging: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceSpec::BestAvailable,
            scheduler: SchedulerPolicy::default(),
            no_kv_cache: false,
            no_prefix_cache: false,
            prefix_cache_entries: 16,
            disable_eos_stop: false,
            throughput_logging: false,
        }
    }
}

/// A single model exposed through `nexo-ai`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModelConfig {
    /// The stable `nexo-core` descriptor surfaced to callers.
    pub descriptor: ModelDescriptor,

    /// The loader configuration used to create the backing runtime pipeline.
    pub loader: ModelLoader,

    /// The optional Hugging Face revision to pin during model loading.
    pub revision: Option<String>,
}

/// Public loader variants supported by `nexo-ai`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelLoader {
    /// Use `mistralrs-core` automatic loader detection for a local path or HF repository.
    Auto(AutoModelLoader),

    /// Load a quantized GGUF model with an explicit weight file list.
    Gguf(GgufModelLoader),
}

/// Loader settings for `mistralrs-core` automatic model detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoModelLoader {
    /// The model identifier or local path understood by `mistralrs-core`.
    pub model_id: String,

    /// Optional UQFF artifact filenames or paths. When omitted, local `.uqff` files are discovered.
    #[serde(default)]
    pub from_uqff: Option<Vec<PathBuf>>,

    /// An optional local `tokenizer.json` path used instead of remote metadata.
    pub tokenizer_json: Option<PathBuf>,

    /// An optional local chat template path forwarded to the loader.
    pub chat_template: Option<PathBuf>,

    /// An optional explicit Jinja template path forwarded to the loader.
    pub jinja_explicit: Option<PathBuf>,

    /// The preferred model data type for model loading.
    pub dtype: ModelDataType,

    /// An optional explicit Hugging Face cache directory.
    pub hf_cache_path: Option<PathBuf>,
}

/// Loader settings for GGUF-based models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufModelLoader {
    /// The optional tokenizer or chat-template model source.
    pub tokenizer_model_id: Option<String>,

    /// The GGUF model identifier or local path containing the quantized weights.
    pub quantized_model_id: String,

    /// The GGUF file names to load from the selected source.
    pub quantized_filenames: Vec<String>,

    /// An optional local chat template path forwarded to the loader.
    pub chat_template: Option<PathBuf>,

    /// An optional explicit Jinja template path forwarded to the loader.
    pub jinja_explicit: Option<PathBuf>,

    /// The preferred activation data type for the GGUF pipeline.
    pub dtype: ModelDataType,
}

/// The public data-type choices supported by `nexo-ai`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDataType {
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

/// The device policy used when loading models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceSpec {
    /// Use the best device supported by the current build and platform.
    #[default]
    BestAvailable,

    /// Force CPU execution.
    Cpu,

    /// Prefer Apple's Metal backend when the crate is built with the `metal` feature.
    Metal,
}

/// The scheduler policy used for multi-sequence generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchedulerPolicy {
    /// Run up to a fixed number of concurrent sequences.
    Fixed {
        /// The maximum number of concurrently running sequences.
        max_running_sequences: NonZeroUsize,
    },
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self::Fixed {
            max_running_sequences: NonZeroUsize::MIN,
        }
    }
}

/// Returns the default configuration path used by `nexo-ai` CLI-oriented helpers.
pub fn default_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-ai.toml")
}
