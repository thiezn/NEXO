use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Gateway configuration, stored at `~/.nexo/nexo-gateway.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewayConfig {
    /// Host interface to bind the WebSocket server to.
    pub host: String,
    /// TCP port to bind the WebSocket server to.
    pub port: u16,
    /// Default log verbosity for the gateway process.
    pub log_level: String,
    /// Interval between emitted tick events, in milliseconds.
    pub tick_interval_ms: u64,
    /// SQLite database path for persistent gateway state.
    pub db_path: String,
    /// Root storage directory used by the gateway.
    pub storage_root: String,
    /// Shared WebSocket authentication token.
    pub auth_token: String,
    /// Path to the git-backed nexo-storage repository.
    pub nexo_storage_path: String,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 6969,
            log_level: "info".to_string(),
            tick_interval_ms: 15000,
            db_path: "~/.nexo/storage/relational/gateway.db".to_string(),
            storage_root: "~/.nexo/storage".to_string(),
            auth_token: nexo_ws_schema::AUTH_TOKEN.to_string(),
            nexo_storage_path: "~/.nexo/nexo-storage".to_string(),
        }
    }
}

impl GatewayConfig {
    /// Return the default path of the persisted gateway config file.
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("nexo-gateway.toml")
    }

    /// Load the gateway configuration, creating it with defaults when missing.
    pub fn load() -> cli_helpers::Result<Self> {
        cli_helpers::config::load_or_create(&Self::config_path())
    }

    /// Persist the gateway configuration to disk.
    pub fn save(&self) -> cli_helpers::Result {
        cli_helpers::config::save(self, &Self::config_path())
    }

    /// Format the bind host and port as a socket address string.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn default_config_values() {
        let config = GatewayConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 6969);
        assert_eq!(config.tick_interval_ms, 15000);
        assert_eq!(config.auth_token, nexo_ws_schema::AUTH_TOKEN);
    }

    #[test]
    fn bind_addr_formatting() {
        let config = GatewayConfig::default();
        assert_eq!(config.bind_addr(), "0.0.0.0:6969");
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = GatewayConfig {
            host: "0.0.0.0".into(),
            port: 8080,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: GatewayConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.host, "0.0.0.0");
        assert_eq!(decoded.port, 8080);
    }
}
