//! WebSocket protocol schema types shared by gateway, node, and client crates.

/// Error payload types used in response frames.
pub mod error;
/// Server-pushed event payloads.
// pub mod events;
/// Request/response/event frame envelope definitions.
pub mod frame;
/// Protocol method enums and request/response payload structs.
// pub mod methods;
/// The message protocol between gateway, client and nodes.
///
/// Contains all Requests, Responses and events possible between the different components.
pub mod protocol;
/// Schema generation helpers for protocol docs and tooling.
pub mod schema;

// pub use events::{
//     CronPayload, EventKind, HeartbeatPayload, MessagePayload, PresencePayload, RunEventPayload,
//     SessionClosedPayload, ShutdownPayload, TickPayload,
// };
pub use frame::Frame;

pub use error::{Error, Result};
pub use nexo_core::message::{ContentPart, ConversationMessage, MessageRole};
pub use nexo_core::model::ModelDefinition;
pub use nexo_core::tools::{ToolCall, ToolDefinition};
pub use nexo_core::{NexoNodeMetrics, ReasoningEffort, ReasoningSettings, ThinkingMode};
pub use protocol::{
    CancelRequest, ExecuteToolEvent, GatewayToNodeMessage, GatewayToUserMessage, InferenceRunEvent,
    LoadModelEvent, NexoEvent, NexoResponse, NodeToGatewayMessage, UnloadModelEvent,
    UserToGatewayMessage,
};
pub use schema::{SchemaSection, generate_schema, schema_json};

/// The protocol version this crate implements.
pub const PROTOCOL_VERSION: u32 = 1;

/// The expected auth header value.
pub const AUTH_TOKEN: &str = "Tm90U29TM2N1cmU=";

/// The HTTP header name for auth.
pub const AUTH_HEADER: &str = "X-NEXO-AUTH";
