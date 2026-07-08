use crate::NexoClient;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::str::FromStr;
use strum::ParseError;
use uuid::Uuid;

/// Unique live peer identifier derived from stable client and device identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(into = "String", try_from = "String")]
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
        write!(f, "{}:{}", self.client_id, self.device_id)
    }
}

impl From<PeerId> for String {
    fn from(peer_id: PeerId) -> Self {
        peer_id.to_string()
    }
}

impl TryFrom<String> for PeerId {
    type Error = ParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl FromStr for PeerId {
    type Err = ParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let Some((client_id, device_id)) = value.split_once(':') else {
            return Err(ParseError::VariantNotFound);
        };

        let client_id = Uuid::parse_str(client_id).map_err(|_| ParseError::VariantNotFound)?;
        let device_id = Uuid::parse_str(device_id).map_err(|_| ParseError::VariantNotFound)?;

        Ok(Self::new(client_id, device_id))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn serializes_to_string() {
        let peer_id = PeerId::new(Uuid::nil(), Uuid::max());

        assert_eq!(
            serde_json::to_string(&peer_id).unwrap(),
            format!("\"{}\"", peer_id)
        );
    }

    #[test]
    fn deserializes_from_string() {
        let expected = PeerId::new(Uuid::nil(), Uuid::max());
        let json = format!("\"{}\"", expected);

        assert_eq!(serde_json::from_str::<PeerId>(&json).unwrap(), expected);
    }
}
