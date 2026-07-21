//! Request, response, streaming, and usage types for inference operations.

/// Structured inference failure types.
pub mod errors;
/// Inference and model lifecycle event payloads shared across runtimes and transports.
pub mod events;
/// Completion finish reason types.
pub mod finish;
/// Inference request variants and request payloads.
pub mod intent;
/// Inference execution metadata.
pub mod meta;
/// Inference stream ordering types.
pub mod ordering;
/// Inference final result variants.
pub mod output;
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

/// Inference compaction request types.
pub mod compaction;

/// Inference request types.
pub mod request;

pub use compaction::CompactionRequest;
pub use errors::{InferenceErrorCode, InferenceFailure, Retryability};
pub use events::{InferenceRunEvent, LoadModelEvent, UnloadModelEvent};
pub use finish::FinishReason;
pub use intent::{InferenceIntent, InferenceOperation, InferenceOperationKind};
pub use meta::InferenceMeta;
pub use ordering::{ArtifactIndex, OutputOffsetBytes, StreamSeq};
pub use output::InferenceOutput;
pub use request::InferenceRequest;
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
    InferenceFailed, InferenceOutputDelta, InferenceProgress, InferenceStarted, InferenceUpdate,
    MultiModalDelta, SpeechDelta, TokenizationDelta,
};
pub use usage::{PerformanceMetrics, TokenUsage};
