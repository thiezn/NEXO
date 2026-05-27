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

pub use errors::{InferenceErrorCode, InferenceFailure, Retryability};
pub use finish::FinishReason;
pub use request::{
    AudioFormat, DetokenizationRequest, EmbedRequest, GenerationPromptPolicy,
    ImageGenerationRequest, ImageGenerationSize, InferenceRequest, SpecialTokenPolicy,
    SpeechGenerationRequest, TokenizationInput, TokenizationRequest,
};
pub use response::{
    DetokenizationResponse, EmbeddingResponse, EmbeddingVector, GenerateChunk, GenerateCompleted,
    GenerateStarted, ImageGenerationResponse, InferenceResponse, SpeechGenerationResponse,
    TokenizationResponse,
};
pub use sampling::{OutputConstraint, SamplingConfig, StreamingMode};
pub use stream::GenerateDelta;
pub use stream::InferenceStream;
pub use usage::{PerformanceMetrics, TokenUsage};
