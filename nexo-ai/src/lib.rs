//! Library-first inference adapters that bridge `nexo-core` and configured runtimes.

#![forbid(unsafe_code)]

/// Helpers for turning local model manifests into runtime configs.
pub mod catalog;
/// Inference engine implementation and configuration.
pub mod engine;

/// Crate-local error and result types.
pub mod error;

pub use catalog::{ModelCatalog, ModelManifest};

pub use engine::InferenceEngine;
pub use engine::any_tts::AnyTtsModelConfig;
pub use engine::config::default_config_path;
pub use engine::config::{
    DeviceSpec, InferenceEngineConfig, ModelDataType, ModelRuntimeImplementation,
    RegisteredModelConfig, RuntimeConfig, RuntimeImplementation, SchedulerPolicy,
};
pub use error::{Error, Result};
