//! WebSocket protocol schema types shared by gateway, node, and client crates.

/// Connect handshake payloads and policy types.
pub mod connect;
/// Error payload types used in response frames.
pub mod error;
/// Server-pushed event payloads.
pub mod events;
/// Request/response/event frame envelope definitions.
pub mod frame;
/// Protocol method enums and request/response payload structs.
pub mod methods;
/// Schema generation helpers for protocol docs and tooling.
pub mod schema;
/// Shared handshake and identity primitives.
pub mod types;

// Re-exports
pub use connect::{ConnectParams, HelloOk, Policy};
pub use error::{ErrorPayload, WsError};
pub use events::{
    CronPayload, EventKind, HeartbeatPayload, MessagePayload, PresencePayload, RunEventPayload,
    SessionClosedPayload, ShutdownPayload, TickPayload,
};
pub use frame::Frame;
pub use methods::PromptCollection;
pub use methods::{
    AudioAnalyzeParams, AudioAnalyzeResponse, AudioGenerateParams, AudioGenerateResponse,
    CronCreateParams, CronCreateResponse, CronDeleteParams, CronDeleteResponse, CronEntry,
    CronListParams, CronListResponse, GeneratedImagePayload, HealthParams, HealthResponse,
    ImageAnalyzeParams, ImageAnalyzeResponse, ImageGenerateParams, ImageGenerateResponse, Method,
    ModelLoadParams, ModelLoadResponse, ModelStatusParams, ModelUnloadParams, ModelUnloadResponse,
    PromptCollectionCreateParams, PromptCollectionCreateResponse, PromptCollectionDeleteParams,
    PromptCollectionDeleteResponse, PromptCollectionListParams, PromptCollectionListResponse,
    PromptDocument, PromptDocumentCreateParams, PromptDocumentCreateResponse,
    PromptDocumentDeleteParams, PromptDocumentDeleteResponse, PromptDocumentEntry,
    PromptDocumentListParams, PromptDocumentListResponse, RunInstructionsAppendParams,
    RunInstructionsAppendResponse, RunRoundRequest, RunRoundResponse, RunRoundToolCall,
    RunStartParams, RunStartResponse, RunStatus, RunStopParams, RunStopResponse, SendParams,
    SendResponse, SessionClearParams, SessionClearResponse, SessionCreateParams,
    SessionCreateResponse, SessionEntry, SessionGetParams, SessionGetResponse, SessionListParams,
    SessionListResponse, StatusParams, StatusResponse, SystemPresenceParams, SystemPrompt,
    ToolEntry, ToolSpecEntry, ToolsCatalogParams, ToolsCatalogResponse, ToolsExecuteParams,
    ToolsExecuteResponse, ToolsRegisterParams, ToolsRegisterResponse,
};
pub use nexo_core::message::{ContentPart, ConversationMessage, MessageRole, TextPart};
pub use nexo_core::model::ModelDescriptor;
pub use nexo_core::tools::{ToolCall, ToolDefinition};
pub use nexo_core::{ReasoningEffort, ReasoningSettings, ThinkingMode};
pub use schema::{SchemaSection, generate_schema, schema_json};
pub use types::{ClientInfo, ConnectionRole, DeviceInfo, Platform, Scope};

/// The protocol version this crate implements.
pub const PROTOCOL_VERSION: u32 = 3;

/// The expected auth header value.
pub const AUTH_TOKEN: &str = "Tm90U29TM2N1cmU=";

/// The HTTP header name for auth.
pub const AUTH_HEADER: &str = "X-NEXO-AUTH";
