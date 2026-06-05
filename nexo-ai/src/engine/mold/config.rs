use serde::{Deserialize, Serialize};

/// The load strategy used by the mold runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoldLoadStrategy {
    /// Keep components hot in memory after loading.
    Eager,

    /// Load components in stages to reduce peak memory use.
    #[default]
    Sequential,
}

/// Runtime defaults for the mold integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MoldRuntimeConfig {
    /// The GPU ordinal to target when the runtime selects a device.
    pub gpu_ordinal: usize,

    /// The staged or eager loading policy.
    pub load_strategy: MoldLoadStrategy,

    /// Force the Flux.2 transformer offload path when supported.
    pub offload: bool,

    /// Optional Qwen3 text-encoder variant hint forwarded to mold.
    pub qwen3_variant: Option<String>,
}

impl Default for MoldRuntimeConfig {
    fn default() -> Self {
        Self {
            gpu_ordinal: 0,
            load_strategy: MoldLoadStrategy::Sequential,
            offload: false,
            qwen3_variant: None,
        }
    }
}

/// mold-specific configuration for a model binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoldModelConfig {
    /// The loader configuration used to create the backing runtime pipeline.
    pub loader: MoldLoader,
}

/// Loader variants supported by the mold integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MoldLoader {
    /// Load a FLUX.2 image-generation model from local NEXO storage.
    Flux2(MoldFlux2Loader),
}

/// Loader settings for FLUX.2 image-generation models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoldFlux2Loader {
    /// The local model identifier under NEXO's model store.
    pub model_id: String,
}
