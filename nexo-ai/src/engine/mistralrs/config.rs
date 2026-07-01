use serde::{Deserialize, Serialize};

/// Runtime defaults for the Mistral.rs integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MistralRsRuntimeConfig {
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
