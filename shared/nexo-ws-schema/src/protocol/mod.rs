//! The NEXO WebSocket protocol schema.
//!
//! This crate defines the schema for the NEXO WebSocket protocol, including the messages that can be sent between clients, gateways, and nodes. It provides a structured way to represent the different types of messages and events that can occur in the NEXO system.

/// The messages that can be sent from a client to a gateway.
pub mod user_to_gateway;

/// The messages that can be sent from a gateway to a node.
pub mod gateway_to_user;

/// The messages that can be sent from a gateway to a node.
pub mod gateway_to_node;

/// Message definitions used by the gateway, nodes and clients for communication.
pub mod messages;

/// The messages that can be sent from a node to a gateway.
pub mod node_to_gateway;

pub use gateway_to_node::GatewayToNodeMessage;
pub use gateway_to_user::GatewayToUserMessage;
pub use node_to_gateway::NodeToGatewayMessage;
pub use user_to_gateway::UserToGatewayMessage;

pub use messages::base::{NexoEvent, NexoResponse};
pub use messages::control::CancelRequest;
pub use messages::inference::{InferenceEvent, LoadModelEvent, UnloadModelEvent};
pub use messages::tools::ExecuteToolEvent;
