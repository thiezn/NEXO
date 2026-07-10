use super::{ClientInfo, DeviceInfo, ProtocolInfo, Scope};
use crate::PeerId;
use crate::ToolDefinition;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[cfg(feature = "sqlx")]
use sqlx::sqlite::SqliteRow;
#[cfg(feature = "sqlx")]
use sqlx::{FromRow, Row};
#[cfg(feature = "sqlx")]
use std::io;

/// A single active User in the NexoGateway
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct User {
    /// Unique identifier for this user, derived from stable client and device identifiers.
    id: PeerId,

    /// Tool definitions exposed by this node.
    #[serde(default)]
    tools: HashSet<ToolDefinition>,

    /// Connected at
    connected_at: chrono::DateTime<chrono::Utc>,
}

impl User {
    /// Initialize a new user with the given peer ID and tools.
    pub fn new(id: PeerId, tools: HashSet<ToolDefinition>) -> Self {
        let connected_at = chrono::Utc::now();
        Self {
            id,
            tools,
            connected_at,
        }
    }

    /// Build a user from the given user properties.
    pub fn from_properties(properties: &UserProperties) -> Self {
        let id = PeerId::new(properties.client().id, properties.device().id);
        let tools = properties.tools().iter().cloned().collect();
        Self::new(id, tools)
    }

    /// Get the unique identifier for this user.
    pub fn id(&self) -> PeerId {
        self.id
    }

    /// Get the set of tool definitions exposed by this user.
    pub fn tools(&self) -> &HashSet<ToolDefinition> {
        &self.tools
    }

    /// Get the timestamp when this user connected.
    pub fn connected_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.connected_at
    }
}

#[cfg(feature = "sqlx")]
impl<'r> FromRow<'r, SqliteRow> for User {
    fn from_row(row: &SqliteRow) -> Result<Self, sqlx::Error> {
        let client_id: String = row.try_get("client_id")?;
        let device_id: String = row.try_get("device_id")?;
        let tools_json: String = row.try_get("tools_json")?;
        let connected_at: chrono::DateTime<chrono::Utc> = row.try_get("connected_at")?;

        Ok(Self {
            id: decode_peer_id(client_id, device_id)?,
            tools: serde_json::from_str(&tools_json).map_err(|error| {
                sqlx::Error::Decode(Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid tools_json '{}': {}", tools_json, error),
                )))
            })?,
            connected_at,
        })
    }
}

#[cfg(feature = "sqlx")]
fn decode_peer_id(client_id: String, device_id: String) -> Result<PeerId, sqlx::Error> {
    let client_id = uuid::Uuid::parse_str(&client_id).map_err(|error| {
        sqlx::Error::Decode(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid client_id '{}': {}", client_id, error),
        )))
    })?;
    let device_id = uuid::Uuid::parse_str(&device_id).map_err(|error| {
        sqlx::Error::Decode(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid device_id '{}': {}", device_id, error),
        )))
    })?;
    Ok(PeerId::new(client_id, device_id))
}

/// Persisted configuration and handshake identity for a user-facing client.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct UserProperties {
    /// WebSocket URL of the gateway this user client connects to.
    gateway_url: String,

    /// Shared gateway authentication token.
    auth_token: String,

    /// Delay between reconnect attempts after a gateway disconnect.
    reconnect_interval_ms: u64,

    /// Protocol negotiation metadata sent during connect.
    protocol: ProtocolInfo,

    /// Client identity metadata.
    #[serde(default)]
    client: ClientInfo,

    /// Device identity used for pairing.
    #[serde(default)]
    device: DeviceInfo,

    /// Requested authorization scopes.
    #[serde(default)]
    scopes: Vec<Scope>,

    /// Tools exposed by this user client, if any.
    #[serde(default)]
    tools: Vec<ToolDefinition>,
}

impl UserProperties {
    /// Start building user properties with explicit identity and auth.
    pub fn builder(
        client: ClientInfo,
        device: DeviceInfo,
        auth_token: impl Into<String>,
    ) -> UserPropertiesBuilder {
        UserPropertiesBuilder::new(client, device, auth_token)
    }

    /// Build user properties with default connection settings.
    pub fn new(client: ClientInfo, device: DeviceInfo, auth_token: impl Into<String>) -> Self {
        Self::builder(client, device, auth_token).build()
    }

    /// Return a builder initialized from these properties.
    pub fn into_builder(self) -> UserPropertiesBuilder {
        UserPropertiesBuilder {
            gateway_url: self.gateway_url,
            auth_token: self.auth_token,
            reconnect_interval_ms: self.reconnect_interval_ms,
            client: self.client,
            device: self.device,
            scopes: self.scopes,
            tools: self.tools,
        }
    }

    /// WebSocket URL of the gateway this user client connects to.
    pub fn gateway_url(&self) -> &str {
        &self.gateway_url
    }

    /// Shared gateway authentication token.
    pub fn auth_token(&self) -> &str {
        &self.auth_token
    }

    /// Delay between reconnect attempts after a gateway disconnect.
    pub fn reconnect_interval_ms(&self) -> u64 {
        self.reconnect_interval_ms
    }

    /// Protocol negotiation metadata sent during connect.
    pub fn protocol(&self) -> &ProtocolInfo {
        &self.protocol
    }

    /// Client identity metadata.
    pub fn client(&self) -> &ClientInfo {
        &self.client
    }

    /// Device identity used for pairing.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Requested authorization scopes.
    pub fn scopes(&self) -> &[Scope] {
        &self.scopes
    }

    /// Tools exposed by this user client, if any.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }
}

impl Default for UserProperties {
    fn default() -> Self {
        Self::new(ClientInfo::default(), DeviceInfo::default(), "")
    }
}

/// Builder for [`UserProperties`].
#[derive(Debug, Clone)]
pub struct UserPropertiesBuilder {
    gateway_url: String,
    auth_token: String,
    reconnect_interval_ms: u64,
    client: ClientInfo,
    device: DeviceInfo,
    scopes: Vec<Scope>,
    tools: Vec<ToolDefinition>,
}

impl UserPropertiesBuilder {
    /// Create a user properties builder with required identity and auth values.
    pub fn new(client: ClientInfo, device: DeviceInfo, auth_token: impl Into<String>) -> Self {
        Self {
            gateway_url: "ws://127.0.0.1:6969".to_string(),
            auth_token: auth_token.into(),
            reconnect_interval_ms: 5000,
            client,
            device,
            scopes: vec![Scope::UserRead, Scope::UserWrite],
            tools: Vec::new(),
        }
    }

    /// Set the gateway URL.
    pub fn gateway_url(mut self, gateway_url: impl Into<String>) -> Self {
        self.gateway_url = gateway_url.into();
        self
    }

    /// Set the reconnect interval in milliseconds.
    pub fn reconnect_interval_ms(mut self, reconnect_interval_ms: u64) -> Self {
        self.reconnect_interval_ms = reconnect_interval_ms;
        self
    }

    /// Set requested authorization scopes.
    pub fn scopes(mut self, scopes: Vec<Scope>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Set tools exposed by this user client.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Build complete user properties.
    pub fn build(self) -> UserProperties {
        let protocol = ProtocolInfo::new_client(&self.client);
        UserProperties {
            gateway_url: self.gateway_url,
            auth_token: self.auth_token,
            reconnect_interval_ms: self.reconnect_interval_ms,
            protocol,
            client: self.client,
            device: self.device,
            scopes: self.scopes,
            tools: self.tools,
        }
    }
}
