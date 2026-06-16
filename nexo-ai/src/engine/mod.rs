//! Inference Engine is responsible for managing the lifecycle of inference runtimes, handling inference requests,
//! and orchestrating the execution of models across different backends.

pub mod inference_engine;
pub use inference_engine::InferenceEngine;

/// Backing runtime integrations used by the engine.
pub(crate) mod mistralrs;
pub(crate) use mistralrs::MistralRsRuntime;

pub(crate) mod mold;
pub(crate) use mold::MoldRuntime;

pub(crate) mod any_tts;
pub(crate) use any_tts::AnyTtsRuntime;
