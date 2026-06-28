use crate::ModelId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Properties of a Nexo Web Socket client (client or node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum ClientKind {
    /// Properties of a Nexo Web Socket client User.
    User(UserProperties),

    /// Properties of a Nexo Web Socket client Node.
    Node(NodeProperties),
}

impl ClientKind {
    /// Create a new `ClientKind` for a node-role client.
    pub fn new_node(
        client_id: &str,
        version: &str,
        platform: Platform,
        device_id: &str,
        capabilities: Vec<String>,
        commands: Vec<String>,
        models: Vec<ModelId>,
    ) -> Self {
        ClientKind::Node(NodeProperties {
            min_protocol: 1,
            max_protocol: 1,
            client: ClientInfo {
                id: client_id.to_string(),
                version: version.to_string(),
            },
            capabilities,
            commands,
            models,
            user_agent: format!("NEXO-NODE-{client_id}/{version}"),
            device: DeviceInfo {
                id: device_id.to_string(),
                platform,
            },
        })
    }

    /// Create a new `ClientKind` for a user-role client.
    pub fn new_user(client_id: &str, version: &str, device_id: &str, platform: Platform) -> Self {
        ClientKind::User(UserProperties {
            min_protocol: 1,
            max_protocol: 1,
            client: ClientInfo {
                id: client_id.to_string(),
                version: version.to_string(),
            },
            scopes: vec![Scope::UserRead, Scope::UserWrite],
            user_agent: format!("NEXO-USER-{client_id}/{version}"),
            device: DeviceInfo {
                id: device_id.to_string(),
                platform,
            },
        })
    }
}

/// Properties of a Nexo Web Socket client User.
///
/// Used during connect handshake to describe the client to the gateway.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct UserProperties {
    /// Minimum protocol version supported by the client.
    pub min_protocol: u32,

    /// Maximum protocol version supported by the client.
    pub max_protocol: u32,

    /// Client identity metadata.
    pub client: ClientInfo,

    #[serde(default)]
    /// Requested authorization scopes
    pub scopes: Vec<Scope>,

    /// user-agent style client descriptor.
    pub user_agent: String,

    /// Device identity used for pairing.
    pub device: DeviceInfo,
}

/// Properties of a Nexo Web Socket client Node.
///
/// Used during connect handshake to describe the client to the gateway.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct NodeProperties {
    /// Minimum protocol version supported by the client.
    pub min_protocol: u32,

    /// Maximum protocol version supported by the client.
    pub max_protocol: u32,

    /// Client identity metadata.
    pub client: ClientInfo,

    #[serde(default)]
    /// Node capability identifiers exposed to the gateway.
    pub capabilities: Vec<String>,

    #[serde(default)]
    /// Command identifiers exposed by the node.
    pub commands: Vec<String>,

    /// Model IDs available on disk for this node. Empty for user clients.
    #[serde(default)]
    pub models: Vec<ModelId>,

    /// user-agent style client descriptor.
    pub user_agent: String,

    /// Device identity used for pairing.
    pub device: DeviceInfo,
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
    pub id: String,

    /// Client version string.
    pub version: String,
}

/// Stable device identity for pairing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable paired-device identifier.
    pub id: String,

    /// Hardware Platform.
    pub platform: Platform,
}
