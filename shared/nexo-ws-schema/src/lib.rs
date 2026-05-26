pub mod connect;
pub mod error;
pub mod events;
pub mod frame;
pub mod methods;
pub mod schema;
pub mod types;

// Re-exports
pub use connect::{ConnectParams, HelloOk, Policy};
pub use error::{ErrorPayload, WsError};
pub use events::{
    CronPayload, EventKind, HeartbeatPayload, MessagePayload, PresencePayload, RunEventPayload,
    ShutdownPayload, TickPayload,
};
pub use frame::Frame;
pub use methods::{
    PromptCollectionCreateParams, PromptCollectionCreateResponse,
    PromptCollectionDeleteParams, PromptCollectionDeleteResponse, PromptCollectionListParams,
    PromptCollectionListResponse, PromptDocumentCreateParams, PromptDocumentCreateResponse,
    PromptDocumentDeleteParams, PromptDocumentDeleteResponse, PromptDocumentEntry,
    PromptDocumentListParams, PromptDocumentListResponse, RunInstructionsAppendParams,
    RunInstructionsAppendResponse, RunRoundRequest, RunRoundResponse, RunRoundToolCall,
    RunStartParams, RunStartResponse, RunStatus, RunStopParams, RunStopResponse,
    CronCreateParams, CronCreateResponse, CronDeleteParams, CronDeleteResponse, CronEntry,
    CronListParams, CronListResponse, HealthParams, HealthResponse, ImageAnalyzeParams,
    ImageAnalyzeResponse, Method, ModelLoadParams, ModelLoadResponse, ModelStatusParams,
    ModelUnloadParams, ModelUnloadResponse, SendParams, SendResponse,
    SessionClearParams, SessionClearResponse, SessionCreateParams, SessionCreateResponse,
    SessionEntry, SessionGetParams, SessionGetResponse, SessionListParams, SessionListResponse,
    StatusParams, StatusResponse, SystemPresenceParams, ToolEntry, ToolSpecEntry,
    ToolsCatalogParams, ToolsCatalogResponse, ToolsExecuteParams, ToolsExecuteResponse,
    ToolsRegisterParams, ToolsRegisterResponse,
};
pub use nexo_spec::message::{MessageRole, TranscriptMessage};
pub use nexo_spec::model::{LoadedModelInfo, ModelCategory};
pub use nexo_spec::prompt::{PromptCollection, PromptDocument, SystemPrompt};
pub use nexo_spec::transcript::{TranscriptEntry, TranscriptEntryKind};
pub use schema::{SchemaSection, generate_schema, schema_json};
pub use types::{ClientInfo, ConnectionRole, DeviceInfo, Platform, Scope};

/// The protocol version this crate implements.
pub const PROTOCOL_VERSION: u32 = 3;

/// The expected auth header value.
pub const AUTH_TOKEN: &str = "Tm90U29TM2N1cmU=";

/// The HTTP header name for auth.
pub const AUTH_HEADER: &str = "X-NEXO-AUTH";
