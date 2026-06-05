//! Node configuration types and persistence helpers.

use nexo_ai::RuntimeConfig;
use nexo_core::{ModelCapability, ModelId};
use nexo_ws_schema::Platform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Node configuration, stored at `~/.nexo/nexo-node.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    /// Gateway websocket URL the node connects to.
    pub gateway_url: String,
    /// Stable node identifier advertised during the gateway handshake.
    pub node_id: String,
    /// Node binary version advertised during the gateway handshake.
    pub node_version: String,
    /// Platform metadata advertised during the gateway handshake.
    pub platform: Platform,
    /// Device identifier advertised during the gateway handshake.
    pub device_id: String,
    /// Shared gateway authentication token.
    pub auth_token: String,
    /// Delay between reconnect attempts after a gateway disconnect.
    pub reconnect_interval_ms: u64,
    /// Capabilities that should be covered by loaded models at startup.
    pub startup_capabilities: Vec<ModelCapability>,
    /// Default model for each startup capability. Missing values use catalog selection.
    pub default_models: HashMap<ModelCapability, ModelId>,
    /// Enables forwarding tools to the model runtime for native tool calling.
    pub enable_tool_calling: bool,
    /// Shared runtime settings forwarded to `nexo-ai`.
    pub runtime: RuntimeConfig,
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
            startup_capabilities: vec![ModelCapability::TextGeneration],
            default_models: HashMap::new(),
            enable_tool_calling: true,
            runtime: RuntimeConfig::default(),
        }
    }
}

impl NodeConfig {
    /// Return the default on-disk path for the node configuration file.
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("nexo-node.toml")
    }

    /// Load the node configuration from disk, creating a default file when needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration file cannot be read or written.
    pub fn load() -> cli_helpers::Result<Self> {
        cli_helpers::config::load_or_create(&Self::config_path())
    }

    /// Persist the current node configuration to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration file cannot be written.
    pub fn save(&self) -> cli_helpers::Result {
        cli_helpers::config::save(self, &Self::config_path())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use nexo_ai::RuntimeImplementation;
    use nexo_ai::engine::mistralrs::MistralRsRuntimeConfig;
    use nexo_core::InferenceRuntime;

    use super::*;

    #[test]
    fn default_config_values() {
        let config = NodeConfig::default();
        assert_eq!(config.gateway_url, "ws://127.0.0.1:6969");
        assert_eq!(config.node_id, "nexo-node");
        assert_eq!(config.auth_token, nexo_ws_schema::AUTH_TOKEN);
        assert_eq!(config.reconnect_interval_ms, 5000);
        assert_eq!(
            config.startup_capabilities,
            vec![ModelCapability::TextGeneration]
        );
        assert!(config.default_models.is_empty());
        assert!(config.enable_tool_calling);
        assert_eq!(
            mistral_runtime(&config.runtime).no_kv_cache,
            mistral_runtime(&RuntimeConfig::default()).no_kv_cache
        );
    }

    #[test]
    fn config_roundtrip() {
        let mut config = NodeConfig {
            gateway_url: "ws://10.0.0.1:8080".into(),
            node_id: "test-node".into(),
            ..Default::default()
        };
        config.default_models.insert(
            ModelCapability::TextGeneration,
            ModelId::from("gemma-4-e2b-it-q5"),
        );
        if let Some(RuntimeImplementation::MistralRs(runtime)) = config
            .runtime
            .runtimes
            .iter_mut()
            .find(|runtime| runtime.runtime() == InferenceRuntime::MistralRs)
        {
            runtime.no_kv_cache = true;
        }

        let json = serde_json::to_string(&config).unwrap();
        let decoded: NodeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.gateway_url, "ws://10.0.0.1:8080");
        assert_eq!(decoded.node_id, "test-node");
        assert_eq!(
            decoded
                .default_models
                .get(&ModelCapability::TextGeneration)
                .unwrap(),
            &ModelId::from("gemma-4-e2b-it-q5")
        );
        assert!(mistral_runtime(&decoded.runtime).no_kv_cache);
    }

    #[test]
    fn no_available_models_in_serialized_config() {
        let config = NodeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("available_models"));
    }

    fn mistral_runtime(runtime: &RuntimeConfig) -> &MistralRsRuntimeConfig {
        runtime
            .runtime(InferenceRuntime::MistralRs)
            .and_then(|runtime| match runtime {
                RuntimeImplementation::MistralRs(config) => Some(config),
                _ => None,
            })
            .unwrap()
    }
}
