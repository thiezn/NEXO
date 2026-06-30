//! Request, response, streaming, and usage types for inference operations.

/// Structured inference failure types.
pub mod errors;
/// Inference final result variants.
pub mod final_output;
/// Completion finish reason types.
pub mod finish;
/// Inference execution metadata.
pub mod meta;
/// Inference stream ordering types.
pub mod ordering;
/// Inference request variants and request payloads.
pub mod request;
/// Inference final response payloads.
pub mod responses;
/// Sampling, streaming, and output constraint settings.
pub mod sampling;
/// Stream delta types and boxed stream aliases.
pub mod stream;
/// Inference update and progressive output types.
pub mod update;
/// Token usage and performance metric types.
pub mod usage;

/// Inference request payload types.
pub mod requests;

/// Model selection primitives.
pub mod selection;

/// Inference session types.
pub mod session;

pub use errors::{InferenceErrorCode, InferenceFailure, Retryability};
pub use final_output::InferenceFinal;
pub use finish::FinishReason;
pub use meta::InferenceMeta;
pub use ordering::{ArtifactIndex, OutputOffsetBytes, StreamSeq};
pub use request::{InferenceOperation, InferenceOperationKind, InferenceRequest};
pub use requests::{
    AudioFormat, DetokenizationPayload, EmbedPayload, GenerationPromptPolicy,
    ImageGenerationPayload, ImageGenerationSize, MultiModalPayload, SpecialTokenPolicy,
    SpeechGenerationPayload, SpeechLanguage, TokenizationPayload,
};
pub use responses::{
    DetokenizationResponse, EmbedResponse, EmbeddingVector, GeneratedAudio, GeneratedImage,
    ImageGenerationResponse, MultiModalResponse, SpeechGenerationResponse, TokenizationResponse,
};
pub use sampling::{OutputConstraint, SamplingConfig, StreamingMode};
pub use selection::ModelSelection;
pub use session::{Session, Sessions};
pub use stream::GenerateDelta;
pub use stream::InferenceStream;
pub use update::{
    DetokenizationDelta, EmbeddingDelta, ImageDelta, InferenceCancelled, InferenceCompleted,
    InferenceFailed, InferenceOutput, InferenceProgress, InferenceStarted, InferenceUpdate,
    MultiModalDelta, SpeechDelta, TokenizationDelta,
};
pub use usage::{PerformanceMetrics, TokenUsage};
