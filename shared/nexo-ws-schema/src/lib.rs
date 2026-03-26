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
    AgentEventPayload, CronPayload, EventKind, HeartbeatPayload, PresencePayload, ShutdownPayload,
    TickPayload,
};
pub use frame::Frame;
pub use methods::{
    AgentParams, AgentResponse, HealthParams, HealthResponse, Method, SendParams, SendResponse,
    StatusParams, StatusResponse, SystemPresenceParams, ToolEntry, ToolSpecEntry,
    ToolsCatalogParams, ToolsCatalogResponse, ToolsExecuteParams, ToolsExecuteResponse,
    ToolsRegisterParams, ToolsRegisterResponse,
};
pub use schema::{SchemaSection, generate_schema, schema_json};
pub use types::{ClientInfo, DeviceInfo, Platform, Role, Scope};

/// The protocol version this crate implements.
pub const PROTOCOL_VERSION: u32 = 3;

/// The expected auth header value.
pub const AUTH_TOKEN: &str = "Tm90U29TM2N1cmU=";

/// The HTTP header name for auth.
pub const AUTH_HEADER: &str = "X-NEXO-AUTH";
