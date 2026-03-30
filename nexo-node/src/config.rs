use nexo_ws_schema::Platform;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Node configuration, stored at `~/.nexo/node.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    pub gateway_url: String,
    pub node_id: String,
    pub node_version: String,
    pub platform: Platform,
    pub device_id: String,
    pub auth_token: String,
    pub reconnect_interval_ms: u64,
    /// Model IDs available on this node's local disk, declared to the gateway at connect time.
    pub available_models: Vec<String>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            gateway_url: "ws://127.0.0.1:6969".to_string(),
            node_id: "nexo-node".to_string(),
            node_version: env!("CARGO_PKG_VERSION").to_string(),
            platform: Platform::current(),
            device_id: "default_node_device".to_string(),
            auth_token: nexo_ws_schema::AUTH_TOKEN.to_string(),
            reconnect_interval_ms: 5000,
            available_models: vec![],
        }
    }
}

impl NodeConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("node.toml")
    }

    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn default_config_values() {
        let config = NodeConfig::default();
        assert_eq!(config.gateway_url, "ws://127.0.0.1:6969");
        assert_eq!(config.node_id, "nexo-node");
        assert_eq!(config.auth_token, nexo_ws_schema::AUTH_TOKEN);
        assert_eq!(config.reconnect_interval_ms, 5000);
    }

    #[test]
    fn config_roundtrip() {
        let config = NodeConfig {
            gateway_url: "ws://10.0.0.1:8080".into(),
            node_id: "test-node".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: NodeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.gateway_url, "ws://10.0.0.1:8080");
        assert_eq!(decoded.node_id, "test-node");
    }
}
