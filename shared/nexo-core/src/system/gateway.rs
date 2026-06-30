use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{ClientInfo, ProtocolInfo};

/// Persisted process configuration for the Nexo gateway server.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct GatewayProperties {
    /// Host interface to bind the WebSocket server to.
    host: String,

    /// TCP port to bind the WebSocket server to.
    port: u16,

    /// Protocol metadata for this gateway process.
    protocol: ProtocolInfo,

    /// Gateway process identity metadata.
    #[serde(default)]
    client: ClientInfo,

    /// SQLite database path for persistent gateway state.
    db_path: String,

    /// Root storage directory used by the gateway.
    storage_root: String,

    /// Shared WebSocket authentication token.
    auth_token: String,

    /// Path to the git-backed nexo-storage repository.
    nexo_storage_path: String,
}

impl GatewayProperties {
    /// Start building gateway properties with explicit identity and auth.
    pub fn builder(client: ClientInfo, auth_token: impl Into<String>) -> GatewayPropertiesBuilder {
        GatewayPropertiesBuilder::new(client, auth_token)
    }

    /// Build gateway properties using the default local server settings.
    pub fn new(client: ClientInfo, auth_token: impl Into<String>) -> Self {
        Self::builder(client, auth_token).build()
    }

    /// Return a builder initialized from these properties.
    pub fn into_builder(self) -> GatewayPropertiesBuilder {
        GatewayPropertiesBuilder {
            host: self.host,
            port: self.port,
            client: self.client,
            db_path: self.db_path,
            storage_root: self.storage_root,
            auth_token: self.auth_token,
            nexo_storage_path: self.nexo_storage_path,
        }
    }

    /// Host interface to bind the WebSocket server to.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// TCP port to bind the WebSocket server to.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Protocol metadata for this gateway process.
    pub fn protocol(&self) -> &ProtocolInfo {
        &self.protocol
    }

    /// Gateway process identity metadata.
    pub fn client(&self) -> &ClientInfo {
        &self.client
    }

    /// SQLite database path for persistent gateway state.
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    /// Root storage directory used by the gateway.
    pub fn storage_root(&self) -> &str {
        &self.storage_root
    }

    /// Shared WebSocket authentication token.
    pub fn auth_token(&self) -> &str {
        &self.auth_token
    }

    /// Path to the git-backed nexo-storage repository.
    pub fn nexo_storage_path(&self) -> &str {
        &self.nexo_storage_path
    }

    /// Format the bind host and port as a socket address string.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl Default for GatewayProperties {
    fn default() -> Self {
        Self::new(ClientInfo::default(), "")
    }
}

/// Builder for [`GatewayProperties`].
#[derive(Debug, Clone)]
pub struct GatewayPropertiesBuilder {
    host: String,
    port: u16,
    client: ClientInfo,
    db_path: String,
    storage_root: String,
    auth_token: String,
    nexo_storage_path: String,
}

impl GatewayPropertiesBuilder {
    /// Create a gateway properties builder with required identity and auth values.
    pub fn new(client: ClientInfo, auth_token: impl Into<String>) -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 6969,
            client,
            db_path: "~/.nexo/storage/relational/gateway.db".to_string(),
            storage_root: "~/.nexo/storage".to_string(),
            auth_token: auth_token.into(),
            nexo_storage_path: "~/.nexo/nexo-storage".to_string(),
        }
    }

    /// Set the bind host.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the bind port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the SQLite database path.
    pub fn db_path(mut self, db_path: impl Into<String>) -> Self {
        self.db_path = db_path.into();
        self
    }

    /// Set the storage root path.
    pub fn storage_root(mut self, storage_root: impl Into<String>) -> Self {
        self.storage_root = storage_root.into();
        self
    }

    /// Set the git-backed nexo-storage path.
    pub fn nexo_storage_path(mut self, nexo_storage_path: impl Into<String>) -> Self {
        self.nexo_storage_path = nexo_storage_path.into();
        self
    }

    /// Build complete gateway properties.
    pub fn build(self) -> GatewayProperties {
        let protocol = ProtocolInfo::new_gateway(&self.client);
        GatewayProperties {
            host: self.host,
            port: self.port,
            protocol,
            client: self.client,
            db_path: self.db_path,
            storage_root: self.storage_root,
            auth_token: self.auth_token,
            nexo_storage_path: self.nexo_storage_path,
        }
    }
}
