//! Shared domain types and cross-crate contracts for the Nexo workspace.
//!
//! `nexo-core` defines the stable Rust-side type system used by transport,
//! gateway orchestration, and inference runtime crates. The crate is designed
//! around a single inference request/response surface with first-class support
//! for multimodal inputs, streamed generation, reasoning controls, and tool
//! interactions.

#![forbid(unsafe_code)]

/// Shared utility types such as timestamps and metadata maps.
pub mod common;
/// Shared error and result types used by crate-level traits.
pub mod error;
/// Strongly typed identifiers used across the Nexo workspace.
pub mod ids;
/// Inference request, response, streaming, and usage types.
pub mod inference;
/// Conversation and multimodal message types.
pub mod message;
/// Model descriptors, capabilities, and selection types.
pub mod model;
/// Higher-level run and round lifecycle types for orchestration.
pub mod run;
/// Tool schemas, execution policies, calls, and results.
pub mod tools;

pub use common::{MetadataMap, PageInfo, PageRequest, Timestamp};
pub use error::{Error, Result};
pub use ids::{ModelId, NodeId, RequestId, RoundId, RunId, SessionId, ToolCallId};
pub use inference::{
    AudioFormat, DetokenizationRequest, DetokenizationResponse, EmbedRequest, EmbeddingResponse,
    EmbeddingVector, FinishReason, GenerateChunk, GenerateCompleted, GenerateDelta,
    GenerateStarted, GenerationPromptPolicy, ImageGenerationRequest, ImageGenerationResponse,
    ImageGenerationSize, InferenceErrorCode, InferenceFailure, InferenceRequest, InferenceResponse,
    InferenceStream, OutputConstraint, PerformanceMetrics, Retryability, SamplingConfig,
    SpecialTokenPolicy, SpeechGenerationRequest, SpeechGenerationResponse, SpeechLanguage,
    StreamingMode, TokenUsage, TokenizationInput, TokenizationRequest, TokenizationResponse,
};
pub use message::{
    AudioInput, ContentPart, Conversation, ConversationMessage, ImageInput, MediaSource,
    MessageRole, TextPart, VideoInput,
};
pub use model::{
    InferenceRuntime, ModelCapability, ModelDescriptor, ModelModalities, ModelRegistry,
    ModelRuntimeState, ModelSelection, ReasoningEffort, ReasoningSettings, RoleStrategy,
    SupportedModality, ThinkingMode,
};
pub use run::{
    RoundEvent, RoundStatus, RoundStatusUpdate, RoundSummary, RunEvent, RunStatus, RunStatusUpdate,
    RunSummary,
};
pub use tools::{
    Tool, ToolCall, ToolCallDelta, ToolChoice, ToolDefinition, ToolExecutionConstraints,
    ToolParallelism, ToolRegistry, ToolResult, ToolResultContent, ToolResultStatus,
    ToolSideEffectLevel,
};
