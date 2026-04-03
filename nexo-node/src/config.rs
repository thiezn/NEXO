use nexo_ai::config::{CoordinatorConfig, ModelSettings};
use nexo_ws_schema::Platform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Node configuration, stored at `~/.nexo/nexo-node.toml`.
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
    /// Categories to auto-load at startup (e.g. `["chat", "tool", "image"]`).
    pub startup_categories: Vec<String>,
    /// Default model for each category (e.g. `{"chat": "gemma-4-e4b-it"}`).
    /// Empty = auto-select smallest available model for each category.
    pub default_models: HashMap<String, String>,
    /// Per-model runtime settings (temperature, max_tokens, etc.).
    pub models: HashMap<String, ModelSettings>,
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
            startup_categories: vec!["chat".to_string(), "tool".to_string(), "image".to_string()],
            default_models: HashMap::new(),
            models: HashMap::new(),
        }
    }
}

impl NodeConfig {
    /// Build a `CoordinatorConfig` from this node's model settings.
    pub fn to_coordinator_config(&self) -> CoordinatorConfig {
        CoordinatorConfig {
            active_models: self.default_models.clone(),
            startup_categories: self.startup_categories.clone(),
            models: self.models.clone(),
        }
    }
}

impl NodeConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("nexo-node.toml")
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
        assert_eq!(config.startup_categories, vec!["chat", "tool", "image"]);
        assert!(config.default_models.is_empty());
        assert!(config.models.is_empty());
    }

    #[test]
    fn config_roundtrip() {
        let mut config = NodeConfig {
            gateway_url: "ws://10.0.0.1:8080".into(),
            node_id: "test-node".into(),
            ..Default::default()
        };
        config
            .default_models
            .insert("chat".into(), "gemma-4-e4b-it".into());
        config.models.insert(
            "gemma-4-e4b-it".into(),
            ModelSettings {
                temperature: Some(0.7),
                max_tokens: Some(2048),
                ..Default::default()
            },
        );

        let json = serde_json::to_string(&config).unwrap();
        let decoded: NodeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.gateway_url, "ws://10.0.0.1:8080");
        assert_eq!(decoded.node_id, "test-node");
        assert_eq!(
            decoded.default_models.get("chat").unwrap(),
            "gemma-4-e4b-it"
        );
        assert_eq!(
            decoded.models.get("gemma-4-e4b-it").unwrap().temperature,
            Some(0.7)
        );
    }

    #[test]
    fn to_coordinator_config_maps_fields() {
        let mut config = NodeConfig::default();
        config
            .default_models
            .insert("chat".into(), "test-model".into());
        config.models.insert(
            "test-model".into(),
            ModelSettings {
                temperature: Some(0.5),
                ..Default::default()
            },
        );

        let coord = config.to_coordinator_config();
        assert_eq!(coord.active_models.get("chat").unwrap(), "test-model");
        assert_eq!(coord.model_settings("test-model").temperature, Some(0.5));
        assert_eq!(coord.startup_categories, vec!["chat", "tool", "image"]);
    }

    #[test]
    fn no_available_models_in_serialized_config() {
        let config = NodeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("available_models"));
    }
}
