use nexo_core::{NexoClient, OperationId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Request to connect a Nexo client to the gateway.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ConnectRequest {
    /// Unique identifier for the connect operation.
    pub operation_id: OperationId,

    /// The connecting domain-level client and its advertised properties.
    pub client: NexoClient,
}

impl ConnectRequest {
    /// Build a connect request for a Nexo client.
    ///
    /// # Arguments
    ///
    /// * `client` - The domain-level client payload and properties to register with the gateway.
    pub fn new(client: NexoClient) -> Self {
        Self {
            operation_id: OperationId::new(),
            client,
        }
    }
}

/// Request to gracefully disconnect from the gateway.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DisconnectRequest {
    /// Unique identifier for the disconnect operation.
    pub operation_id: OperationId,
}

impl DisconnectRequest {
    /// Build a disconnect request.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn new() -> Self {
        Self {
            operation_id: OperationId::new(),
        }
    }
}
