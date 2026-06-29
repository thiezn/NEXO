use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{DeviceSpec, ModelDataType};

/// Runtime defaults for the Mistral.rs integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MistralRsRuntimeConfig {
    /// The target device selection policy used for model loading.
    pub device: DeviceSpec,

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

    /// PagedAttention runtime controls.
    pub paged_attention: MistralRsPagedAttentionConfig,
}

impl Default for MistralRsRuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceSpec::BestAvailable,
            no_kv_cache: false,
            no_prefix_cache: true,
            prefix_cache_entries: 16,
            disable_eos_stop: false,
            throughput_logging: false,
            paged_attention: MistralRsPagedAttentionConfig::default(),
        }
    }
}

/// PagedAttention controls for the Mistral.rs runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MistralRsPagedAttentionConfig {
    /// Tri-state mode:
    /// - `auto`: use backend defaults (enabled on CUDA, disabled on Metal/CPU)
    /// - `enabled`: force-enable when supported by the build/backend
    /// - `disabled`: force-disable
    pub mode: MistralRsPagedAttentionMode,

    /// Optional fixed GPU memory budget (MB) for paged KV cache.
    pub gpu_memory_mb: Option<usize>,

    /// Optional GPU memory utilization ratio in `[0, 1]` for paged KV cache.
    pub gpu_memory_utilization: Option<f32>,

    /// Optional context-length target for paged KV cache sizing.
    pub context_size: Option<usize>,

    /// Optional tokens-per-block override (supported values depend on backend).
    pub block_size: Option<usize>,

    /// KV cache dtype policy for paged attention.
    pub cache_type: MistralRsPagedAttentionCacheType,
}

impl Default for MistralRsPagedAttentionConfig {
    fn default() -> Self {
        Self {
            mode: MistralRsPagedAttentionMode::Auto,
            gpu_memory_mb: None,
            gpu_memory_utilization: None,
            context_size: None,
            block_size: None,
            cache_type: MistralRsPagedAttentionCacheType::Auto,
        }
    }
}

/// PagedAttention enablement policy for Mistral.rs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MistralRsPagedAttentionMode {
    /// Use backend defaults (CUDA: enabled, Metal/CPU: disabled).
    #[default]
    Auto,
    /// Force-enable paged attention when supported.
    Enabled,
    /// Force-disable paged attention.
    Disabled,
}

/// PagedAttention KV-cache storage type for Mistral.rs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MistralRsPagedAttentionCacheType {
    /// Let mistral-rs choose the cache dtype.
    #[default]
    Auto,
    /// Force f8e4m3 cache storage.
    F8e4m3,
}

/// Mistral.rs-specific configuration for a model binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsModelConfig {
    /// The loader configuration used to create the backing runtime pipeline.
    pub loader: MistralRsLoader,

    /// The optional Hugging Face revision to pin during model loading.
    pub revision: Option<String>,
}

/// Loader variants supported by the Mistral.rs integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MistralRsLoader {
    /// Use `mistralrs-core` automatic loader detection for a local path or HF repository.
    Auto(MistralRsAutoLoader),

    /// Load a diffusion model for image generation.
    Diffusion(MistralRsDiffusionLoader),

    /// Load a speech synthesis model.
    Speech(MistralRsSpeechLoader),

    /// Load a quantized GGUF model with an explicit weight file list.
    Gguf(MistralRsGgufLoader),
}

/// Loader settings for `mistralrs-core` automatic model detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsAutoLoader {
    /// The model identifier or local path understood by `mistralrs-core`.
    pub model_id: String,

    /// Optional UQFF artifact filenames, paths, or local filename prefixes.
    /// When omitted, local `.uqff` files are discovered.
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

/// Loader settings for diffusion image-generation models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsDiffusionLoader {
    /// The local model identifier under NEXO's model store.
    pub model_id: String,

    /// Prefer the offloaded FLUX loader variant when supported by the runtime.
    #[serde(default)]
    pub offload: bool,

    /// The preferred model data type for model loading.
    pub dtype: ModelDataType,
}

/// Loader settings for speech synthesis models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsSpeechLoader {
    /// The local model identifier under NEXO's model store.
    pub model_id: String,

    /// Optional local DAC model identifier or path.
    pub dac_model_id: Option<String>,

    /// The preferred model data type for model loading.
    pub dtype: ModelDataType,
}

/// Loader settings for GGUF-based models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsGgufLoader {
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
