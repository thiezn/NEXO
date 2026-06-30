use crate::Result;
use nexo_core::{GatewayProperties, NodeProperties, UserProperties};
use nexo_ws_schema::{
    Frame, GatewayToNodeMessage, GatewayToUserMessage, NodeToGatewayMessage, UserToGatewayMessage,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Central coordinator for nexo-gateway, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub struct NexoGateway {
    /// The configuration for the gateway.
    config: GatewayProperties,

    /// The list of connected users.
    users: Vec<UserProperties>,

    /// The list of connected nodes.
    nodes: Vec<NodeProperties>,
}

impl NexoGateway {
    pub fn new(config: GatewayProperties) -> Self {
        Self {
            config,
            users: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub async fn run(&self) -> Result {
        let (node_tx, mut node_rx) = mpsc::channel::<NodeToGatewayMessage>(100);
        let (user_tx, mut user_rx) = mpsc::channel::<UserToGatewayMessage>(100);

        Ok(())
    }
}
