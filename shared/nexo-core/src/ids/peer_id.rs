use crate::NexoClient;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use uuid::Uuid;

/// Unique live peer identifier derived from stable client and device identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PeerId {
    /// Stable client identifier advertised by the peer.
    ///
    /// Multiple clients can share the same device.
    client_id: Uuid,

    /// Stable device identifier advertised by the peer.
    ///
    /// One client can have multiple devices, each with a unique device identifier.
    device_id: Uuid,
}

impl PeerId {
    /// Build a peer identifier from stable client and device identifiers.
    ///
    /// # Arguments
    ///
    /// * `client_id` - Stable client identifier advertised by the peer.
    /// * `device_id` - Stable device identifier advertised by the peer.
    pub fn new(client_id: Uuid, device_id: Uuid) -> Self {
        Self {
            client_id,
            device_id,
        }
    }

    /// Build a peer identifier from a connected Nexo client payload.
    ///
    /// # Arguments
    ///
    /// * `client` - The domain-level client payload received during connect.
    pub fn from_client(client: &NexoClient) -> Self {
        Self::new(client.client().id, client.device().id)
    }

    /// Return the stable client identifier.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn client_id(&self) -> Uuid {
        self.client_id
    }

    /// Return the stable device identifier.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn device_id(&self) -> Uuid {
        self.device_id
    }
}

impl Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.client_id, self.device_id)
    }
}
