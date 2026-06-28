/// The state of the Nexo System.
pub mod state;

/// Metrics related to a Nexo system components
pub mod metrics;

/// Properties of a Nexo Web Socket client (client or node).
pub mod client;

pub use client::{ClientKind, NodeProperties, Platform, UserProperties};
pub use metrics::NexoNodeMetrics;
pub use state::NexoState;
