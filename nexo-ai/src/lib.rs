//! Library-first inference adapters that bridge `nexo-core` and `mistralrs-core`.

#![forbid(unsafe_code)]

/// Helpers for turning local model manifests into runtime configs.
pub mod catalog;
/// Public runtime configuration and model loader configuration types.
pub mod config;
/// Crate-local error and result types.
pub mod error;
mod mapping;
pub mod registry;
mod round;
mod run;
pub mod runtime;

pub use catalog::{downloaded_model_configs, model_config_from_manifest};
pub use config::default_config_path;
pub use config::{
    AutoModelLoader, DeviceSpec, GgufModelLoader, ModelDataType, ModelLoader, NexoAiConfig,
    RegisteredModelConfig, RuntimeConfig, SchedulerPolicy,
};
pub use error::{Error, Result};
pub use registry::StaticModelRegistry;
pub use runtime::{NexoAi, NexoAiBuilder};

pub use nexo_core::{
    InferenceEngine, InferenceRequest, InferenceResponse, InferenceStream, ModelDescriptor,
    ModelId, ModelRegistry, ModelRuntimeState,
};
