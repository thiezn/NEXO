use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, RequestId, RoundId, RunId};
use crate::message::ConversationMessage;

use super::errors::InferenceFailure;
use super::finish::FinishReason;
use super::requests::{GeneratedAudio, GeneratedImage};
use super::stream::GenerateDelta;
use super::usage::{PerformanceMetrics, TokenUsage};

/// The unified response enum returned by inference engines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceResponse {
    /// Signals that a generation request has started.
    GenerationStarted(GenerateStarted),

    /// Emits a streamed generation chunk.
    GenerationChunk(GenerateChunk),

    /// Emits the final generation result.
    GenerationCompleted(GenerateCompleted),

    /// Returns embedding vectors.
    Embeddings(EmbeddingResponse),

    /// Returns one or more generated images.
    Images(ImageGenerationResponse),

    /// Returns generated speech audio.
    Speech(SpeechGenerationResponse),

    /// Returns tokenized output.
    Tokenization(TokenizationResponse),

    /// Returns detokenized text.
    Detokenization(DetokenizationResponse),

    /// Returns a structured inference failure.
    Failure(InferenceFailure),
}

/// Metadata emitted when generation begins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateStarted {
    /// The request identifier associated with the generation.
    pub request_id: RequestId,

    /// The run identifier associated with the generation.
    pub run_id: RunId,

    /// The round identifier associated with the generation.
    pub round_id: RoundId,

    /// The selected model, if one has been resolved.
    pub model_id: ModelId,
}

/// A streamed chunk for a generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateChunk {
    /// The request identifier associated with the chunk.
    pub request_id: RequestId,

    /// The run identifier associated with the chunk.
    pub run_id: RunId,

    /// The round identifier associated with the chunk.
    pub round_id: RoundId,

    /// The selected model.
    pub model_id: ModelId,

    /// The streamed content delta.
    pub delta: GenerateDelta,

    /// Incremental or cumulative token usage, if available.
    pub usage: Option<TokenUsage>,

    /// The finish reason, if this chunk terminates the stream.
    pub finish_reason: Option<FinishReason>,
}

/// The final response for a completed generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateCompleted {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The run identifier associated with the response.
    pub run_id: RunId,

    /// The round identifier associated with the response.
    pub round_id: RoundId,

    /// The selected model.
    pub model_id: ModelId,

    /// The final assistant message.
    pub message: ConversationMessage,

    /// The full reasoning content captured during generation, if requested.
    pub reasoning: Option<String>,

    /// The final reason generation ended.
    pub finish_reason: FinishReason,

    /// Token usage recorded for the completed response.
    pub usage: Option<TokenUsage>,

    /// Optional performance metrics recorded for the response.
    pub performance: Option<PerformanceMetrics>,
}

/// A single embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingVector {
    /// The zero-based order of the vector within the response.
    pub index: usize,

    /// The embedding values.
    pub values: Vec<f32>,
}

/// The response returned for an embedding request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingResponse {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The selected model
    pub model_id: ModelId,

    /// The embedding vectors returned by the runtime.
    pub vectors: Vec<EmbeddingVector>,

    /// Token usage recorded for the embedding operation.
    pub usage: Option<TokenUsage>,
}

/// The response returned for an image generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationResponse {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The selected model
    pub model_id: ModelId,

    /// The generated images.
    pub images: Vec<GeneratedImage>,
}

/// The response returned for a speech generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SpeechGenerationResponse {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The selected model, if one has been resolved.
    pub model_id: ModelId,

    /// The generated speech audio payload.
    pub audio: GeneratedAudio,
}

/// The response returned for a tokenization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenizationResponse {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The resulting token ids.
    pub tokens: Vec<u32>,
}

/// The response returned for a detokenization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DetokenizationResponse {
    /// The request identifier associated with the response.
    pub request_id: RequestId,

    /// The detokenized text.
    pub text: String,
}
