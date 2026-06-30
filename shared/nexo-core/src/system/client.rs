use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{NodeProperties, UserProperties};

/// Default Nexo protocol version used by local constructors.
pub const DEFAULT_PROTOCOL_VERSION: u32 = 1;

/// Domain-level Nexo WebSocket client identity and advertised properties.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, strum::EnumDiscriminants)]
#[strum_discriminants(name(NexoClientKind))]
#[strum_discriminants(vis(pub))]
#[strum_discriminants(doc = "The domain-level kind of Nexo WebSocket client.")]
#[strum_discriminants(derive(
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr
))]
#[strum_discriminants(serde(rename_all = "snake_case"))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
#[serde(tag = "kind", content = "properties", rename_all = "snake_case")]
pub enum NexoClient {
    /// Properties of a Nexo Web Socket client User.
    User(UserProperties),

    /// Properties of a Nexo Web Socket client Node.
    Node(NodeProperties),
}

impl NexoClient {
    /// Return the domain-level kind for this client.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn kind(&self) -> NexoClientKind {
        self.into()
    }

    /// Return the shared gateway authentication token for this connection.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn auth_token(&self) -> &str {
        match self {
            Self::User(properties) => properties.auth_token(),
            Self::Node(properties) => properties.auth_token(),
        }
    }

    /// Return the advertised protocol negotiation metadata.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn protocol(&self) -> &ProtocolInfo {
        match self {
            Self::User(properties) => properties.protocol(),
            Self::Node(properties) => properties.protocol(),
        }
    }

    /// Return the stable client identity metadata.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn client(&self) -> &ClientInfo {
        match self {
            Self::User(properties) => properties.client(),
            Self::Node(properties) => properties.client(),
        }
    }

    /// Return the stable device identity metadata.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn device(&self) -> &DeviceInfo {
        match self {
            Self::User(properties) => properties.device(),
            Self::Node(properties) => properties.device(),
        }
    }
}

/// Protocol negotiation metadata advertised by a Nexo participant.
///
/// This is persisted with client/node properties and sent during the gateway
/// connection handshake.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct ProtocolInfo {
    /// Minimum protocol version supported by the client.
    pub min_protocol: u32,

    /// Maximum protocol version supported by the client.
    pub max_protocol: u32,

    /// user-agent style client descriptor.
    pub user_agent: String,
}

impl ProtocolInfo {
    /// Build protocol metadata for a user-facing client.
    pub fn new_client(client: &ClientInfo) -> Self {
        Self::for_role("USER", client)
    }

    /// Build protocol metadata for a node client.
    pub fn new_node(client: &ClientInfo) -> Self {
        Self::for_role("NODE", client)
    }

    /// Build protocol metadata for a gateway process.
    pub fn new_gateway(client: &ClientInfo) -> Self {
        Self::for_role("GATEWAY", client)
    }

    fn for_role(role: &str, client: &ClientInfo) -> Self {
        Self {
            min_protocol: DEFAULT_PROTOCOL_VERSION,
            max_protocol: DEFAULT_PROTOCOL_VERSION,
            user_agent: format!("NEXO-{role}-{}/{}", client.id, client.version),
        }
    }
}

impl Default for ProtocolInfo {
    fn default() -> Self {
        Self::new_client(&ClientInfo::default())
    }
}

fn generate_uuid() -> Uuid {
    Uuid::now_v7()
}

/// Authorization scopes for user-role connections.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub enum Scope {
    /// Read-only user scope.
    #[serde(rename = "user.read")]
    UserRead,
    /// Read/write user scope.
    #[serde(rename = "user.write")]
    UserWrite,
    /// Administrative user scope.
    #[serde(rename = "user.admin")]
    UserAdmin,
}

/// Platform the client is running on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    /// Apple macOS platform.
    Macos,
    /// Apple iOS platform.
    Ios,
    /// Linux platform.
    Linux,
    /// Microsoft Windows platform.
    Windows,
    /// Mortimmy platform identifier.
    Mortimmy,
}

impl Platform {
    /// Detect the current platform from `std::env::consts::OS`.
    pub fn current() -> Self {
        match std::env::consts::OS {
            "macos" => Self::Macos,
            "ios" => Self::Ios,
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            "mortimmy" => Self::Mortimmy, // TODO: This obviously doesn't work with consts::OS
            _ => Self::Macos,
        }
    }
}

/// Client identity included in the connect handshake.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ClientInfo {
    /// Stable client identifier.
    #[serde(default = "generate_uuid")]
    pub id: Uuid,

    /// Client version string.
    pub version: String,
}

impl ClientInfo {
    /// Build client identity metadata with a generated stable identifier.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            id: generate_uuid(),
            version: version.into(),
        }
    }
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self::new("unknown")
    }
}

/// Stable device identity for pairing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable paired-device identifier.
    #[serde(default = "generate_uuid")]
    pub id: Uuid,

    /// Hardware Platform.
    pub platform: Platform,
}

impl DeviceInfo {
    /// Build device identity metadata with a generated stable identifier.
    pub fn new(platform: Platform) -> Self {
        Self {
            id: generate_uuid(),
            platform,
        }
    }
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self::new(Platform::current())
    }
}
