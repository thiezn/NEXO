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

// /// Higher-level run and round lifecycle types for orchestration.
// This is now handled in the StartInferenceRun sequences
// pub mod run;

/// Tool schemas, execution policies, calls, and results.
pub mod tools;

/// System-level types for Nexo runtime and orchestration.
pub mod system;

pub use common::{PageInfo, PageRequest, Timestamp};
pub use error::{Error, Result};
pub use ids::{
    FrameId, ModelId, NodeId, OperationId, PeerId, RoundId, RunId, SessionId, ToolCallId,
};
pub use inference::{
    ArtifactIndex, AudioFormat, CompactionRequest, DetokenizationDelta, DetokenizationPayload,
    DetokenizationResponse, EmbedPayload, EmbedResponse, EmbeddingDelta, EmbeddingVector,
    FinishReason, GenerateDelta, GeneratedAudio, GenerationPromptPolicy, ImageDelta,
    ImageGenerationPayload, ImageGenerationResponse, ImageGenerationSize, InferenceCancelled,
    InferenceCompleted, InferenceErrorCode, InferenceFailed, InferenceFailure, InferenceIntent,
    InferenceMeta, InferenceOperation, InferenceOperationKind, InferenceOutput,
    InferenceOutputDelta, InferenceProgress, InferenceRequest, InferenceStarted, InferenceStream,
    InferenceUpdate, ModelSelection, MultiModalDelta, MultiModalResponse, OutputConstraint,
    OutputOffsetBytes, PerformanceMetrics, Retryability, SamplingConfig, Session, Sessions,
    SpecialTokenPolicy, SpeechDelta, SpeechGenerationPayload, SpeechGenerationResponse,
    SpeechLanguage, StreamSeq, StreamingMode, TokenUsage, TokenizationDelta, TokenizationPayload,
    TokenizationResponse,
};
pub use message::{
    AudioInput, ContentPart, Conversation, ConversationMessage, ImageInput, MediaSource,
    MessageRole, VideoInput,
};
pub use model::{
    ModelCapability, ModelDefinition, ModelFamily, ModelRuntimeState, ReasoningEffort,
    ReasoningSettings, RoleStrategy, ThinkingMode,
};
// pub use run::{
//     RoundEvent, RoundStatus, RoundStatusUpdate, RoundSummary, RunEvent, RunStatus, RunStatusUpdate,
//     RunSummary,
// };
pub use system::{
    ClientInfo, DeviceInfo, GatewayProperties, NexoClient, NexoClientKind, NexoNodeMetrics,
    NexoState, Node, NodeProperties, Platform, ProtocolInfo, Scope, User, UserProperties,
};
pub use tools::{
    Tool, ToolCall, ToolCallDelta, ToolChoice, ToolDefinition, ToolExecutionConstraints,
    ToolParallelism, ToolRegistry, ToolResult, ToolResultContent, ToolResultStatus,
    ToolSideEffectLevel,
};
