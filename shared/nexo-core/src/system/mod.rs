/// The state of the Nexo System.
pub mod state;

/// Metrics related to a Nexo system components
pub mod metrics;

/// Properties of a Nexo Web Socket client (client or node).
pub mod client;

/// Gateway process configuration.
pub mod gateway;

/// Node configuration and advertised runtime properties.
pub mod node;

/// User client configuration and advertised properties.
pub mod user;

pub use client::{ClientInfo, ClientKind, DeviceInfo, Platform, ProtocolInfo, Scope};
pub use gateway::GatewayProperties;
pub use metrics::NexoNodeMetrics;
pub use node::NodeProperties;
pub use state::NexoState;
pub use user::UserProperties;
