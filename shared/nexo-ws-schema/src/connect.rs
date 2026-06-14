use crate::types::{ClientInfo, ConnectionRole, DeviceInfo, Scope};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `connect` handshake request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct ConnectParams {
    /// Minimum protocol version supported by the client.
    pub min_protocol: u32,
    /// Maximum protocol version supported by the client.
    pub max_protocol: u32,
    /// Client identity metadata.
    pub client: ClientInfo,
    /// Connection role (`user` or `node`).
    pub role: ConnectionRole,
    #[serde(default)]
    /// Requested authorization scopes (typically for user role).
    pub scopes: Vec<Scope>,
    #[serde(default)]
    /// Node capability identifiers exposed to the gateway.
    pub capabilities: Vec<String>,
    #[serde(default)]
    /// Command identifiers exposed by the node.
    pub commands: Vec<String>,
    /// Model IDs available on disk for this node. Empty for user clients.
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional locale hint (for example `en-US`).
    pub locale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional user-agent style client descriptor.
    pub user_agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional stable device identity used for pairing.
    pub device: Option<DeviceInfo>,
}

/// Tick/heartbeat policy sent in the hello-ok response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct Policy {
    /// Tick/heartbeat interval in milliseconds.
    pub tick_interval_ms: u64,
}

/// Successful connect response payload.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HelloOk {
    #[serde(rename = "type")]
    /// Wire payload discriminator, always `hello-ok`.
    pub payload_type: String,
    /// Negotiated protocol version.
    pub protocol: u32,
    /// Server heartbeat/tick policy.
    pub policy: Policy,
}

impl Default for HelloOk {
    fn default() -> Self {
        Self {
            payload_type: "hello-ok".to_string(),
            protocol: crate::PROTOCOL_VERSION,
            policy: Policy {
                tick_interval_ms: 15000,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::types::Platform;

    #[test]
    fn connect_params_user_serialization() {
        let params = ConnectParams {
            min_protocol: 3,
            max_protocol: 3,
            client: ClientInfo {
                id: "cli".into(),
                version: "1.2.3".into(),
                platform: Platform::Macos,
            },
            role: ConnectionRole::User,
            scopes: vec![Scope::UserRead, Scope::UserWrite],
            capabilities: vec![],
            commands: vec![],
            models: vec![],
            locale: Some("en-US".into()),
            user_agent: Some("NEXO-cli/1.2.3".into()),
            device: Some(DeviceInfo {
                id: "device_fingerprint".into(),
            }),
        };

        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["minProtocol"], 3);
        assert_eq!(json["maxProtocol"], 3);
        assert_eq!(json["client"]["id"], "cli");
        assert_eq!(json["client"]["platform"], "macos");
        assert_eq!(json["role"], "user");
        assert_eq!(json["scopes"][0], "user.read");
        assert_eq!(json["locale"], "en-US");
        assert_eq!(json["userAgent"], "NEXO-cli/1.2.3");
        assert_eq!(json["device"]["id"], "device_fingerprint");
    }

    #[test]
    fn connect_params_node_serialization() {
        let params = ConnectParams {
            min_protocol: 3,
            max_protocol: 3,
            client: ClientInfo {
                id: "rust-node".into(),
                version: "1.2.3".into(),
                platform: Platform::Macos,
            },
            role: ConnectionRole::Node,
            scopes: vec![],
            capabilities: vec!["game_extractor".into(), "epub_extractor".into()],
            commands: vec![
                "game_extractor.extract".into(),
                "game_extractor.analyze".into(),
                "epub_extractor.extract".into(),
            ],
            models: vec!["qwen3-30b".into()],
            locale: Some("en-US".into()),
            user_agent: Some("NEXO-rust-node/1.2.3".into()),
            device: Some(DeviceInfo {
                id: "device_fingerprint".into(),
            }),
        };

        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["role"], "node");
        assert_eq!(json["capabilities"][0], "game_extractor");
        assert_eq!(json["commands"][0], "game_extractor.extract");
    }

    #[test]
    fn connect_params_roundtrip() {
        let params = ConnectParams {
            min_protocol: 3,
            max_protocol: 3,
            client: ClientInfo {
                id: "test".into(),
                version: "0.1.0".into(),
                platform: Platform::Linux,
            },
            role: ConnectionRole::User,
            scopes: vec![Scope::UserRead],
            capabilities: vec![],
            commands: vec![],
            models: vec![],
            locale: None,
            user_agent: None,
            device: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: ConnectParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn hello_ok_default() {
        let hello = HelloOk::default();
        assert_eq!(hello.payload_type, "hello-ok");
        assert_eq!(hello.protocol, 3);
        assert_eq!(hello.policy.tick_interval_ms, 15000);
    }

    #[test]
    fn hello_ok_serialization() {
        let hello = HelloOk::default();
        let json = serde_json::to_value(&hello).unwrap();
        assert_eq!(json["type"], "hello-ok");
        assert_eq!(json["protocol"], 3);
        assert_eq!(json["policy"]["tickIntervalMs"], 15000);
    }

    #[test]
    fn hello_ok_roundtrip() {
        let hello = HelloOk::default();
        let json = serde_json::to_string(&hello).unwrap();
        let decoded: HelloOk = serde_json::from_str(&json).unwrap();
        assert_eq!(hello, decoded);
    }
}
