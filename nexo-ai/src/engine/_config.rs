use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use nexo_core::ModelDefinition;
use serde::{Deserialize, Serialize};

use crate::engine::any_tts::AnyTtsModelConfig;
use crate::engine::mistralrs::{MistralRsModelConfig, MistralRsRuntimeConfig};
use crate::engine::mold::{MoldModelConfig, MoldRuntimeConfig};
use crate::{Error, Result};

/// Runtime-default configuration for a concrete inference implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "runtime", rename_all = "snake_case")]
pub enum RuntimeImplementation {
    /// Defaults for the Mistral.rs runtime integration.
    MistralRs(MistralRsRuntimeConfig),

    /// Defaults for the mold runtime integration.
    Mold(MoldRuntimeConfig),
}

impl RuntimeImplementation {
    pub(crate) fn as_mistralrs(&self) -> Option<&MistralRsRuntimeConfig> {
        match self {
            Self::MistralRs(config) => Some(config),
            Self::Mold(_) => None,
        }
    }

    pub(crate) fn as_mold(&self) -> Option<&MoldRuntimeConfig> {
        match self {
            Self::MistralRs(_) => None,
            Self::Mold(config) => Some(config),
        }
    }
}

/// Per-model runtime binding for a concrete inference implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "runtime", rename_all = "snake_case")]
pub enum ModelRuntimeImplementation {
    /// AnyTTS-specific model binding configuration.
    AnyTts(AnyTtsModelConfig),

    /// Mistral.rs-specific model binding configuration.
    MistralRs(MistralRsModelConfig),

    /// mold-specific model binding configuration.
    Mold(MoldModelConfig),
}

impl ModelRuntimeImplementation {
    pub(crate) fn as_mistralrs(&self) -> Option<&MistralRsModelConfig> {
        match self {
            Self::AnyTts(_) => None,
            Self::MistralRs(config) => Some(config),
            Self::Mold(_) => None,
        }
    }

    pub(crate) fn as_mold(&self) -> Option<&MoldModelConfig> {
        match self {
            Self::AnyTts(_) | Self::MistralRs(_) => None,
            Self::Mold(config) => Some(config),
        }
    }
}

/// Serializable crate configuration for a library-first `nexo-ai` runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceEngineConfig {
    ///Scheduler policy shared across configured models.
    pub scheduler: SchedulerPolicy,

    /// The logical models exposed through the local registry and runtime.
    pub models: Vec<ModelDefinition>,
}

impl InferenceEngineConfig {
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
