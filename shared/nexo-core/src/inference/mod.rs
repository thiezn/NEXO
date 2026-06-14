//! Request, response, streaming, and usage types for inference operations.

/// Structured inference failure types.
pub mod errors;
/// Completion finish reason types.
pub mod finish;
/// Inference request variants and request payloads.
pub mod request;
/// Inference response variants and completed payloads.
pub mod response;
/// Sampling, streaming, and output constraint settings.
pub mod sampling;
/// Stream delta types and boxed stream aliases.
pub mod stream;
/// Token usage and performance metric types.
pub mod usage;

/// Inference request payload types.
pub mod requests;

/// Model selection primitives.
pub mod selection;

pub use errors::{InferenceErrorCode, InferenceFailure, Retryability};
pub use finish::FinishReason;
pub use request::{InferenceOperation, InferenceRequest};
pub use requests::{
    AudioFormat, DetokenizationPayload, EmbedPayload, GeneratedImage, GenerationPromptPolicy,
    ImageGenerationPayload, ImageGenerationSize, MultiModalPayload, SpecialTokenPolicy,
    SpeechGenerationPayload, SpeechLanguage, TokenizationPayload,
};
pub use response::{
    DetokenizationResponse, EmbeddingResponse, EmbeddingVector, GenerateChunk, GenerateCompleted,
    GenerateStarted, ImageGenerationResponse, InferenceResponse, SpeechGenerationResponse,
    TokenizationResponse,
};
pub use sampling::{OutputConstraint, SamplingConfig, StreamingMode};
pub use selection::ModelSelection;
pub use stream::GenerateDelta;
pub use stream::InferenceStream;
pub use usage::{PerformanceMetrics, TokenUsage};
