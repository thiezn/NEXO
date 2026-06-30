//! Inference Engine is responsible for managing the lifecycle of inference runtimes, handling inference requests,
//! and orchestrating the execution of models across different backends.

/// The main inference engine that manages runtimes and models.
pub mod inference_engine;
pub use inference_engine::InferenceEngine;
/// Backend runtime for the MistralRS model execution.
pub(crate) mod mistralrs;
pub(crate) use mistralrs::MistralRsRuntime;

/// Backend runtime for the Mold model execution.
pub(crate) mod mold;
pub(crate) use mold::MoldRuntime;

/// Backend runtime for the AnyTTS model execution.
pub(crate) mod any_tts;
pub(crate) use any_tts::AnyTtsRuntime;
