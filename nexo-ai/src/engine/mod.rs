//! Inference Engine is responsible for managing the lifecycle of inference runtimes, handling inference requests,
//! and orchestrating the execution of models across different backends.

/// Engine-level runtime and model configuration types.
// pub mod config;
mod inference_engine;
pub use inference_engine::InferenceEngine;
/// Backing runtime integrations used by the engine.
/// mod any_tts;
pub mod mistralrs;
pub(crate) mod mold;
