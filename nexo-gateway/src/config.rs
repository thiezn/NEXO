use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Gateway configuration, stored at `~/.nexo/gateway.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub tick_interval_ms: u64,
    pub db_path: String,
    pub storage_root: String,
    pub auth_token: String,
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
        }
    }
}

impl GatewayConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("gateway.toml")
    }

    pub fn load() -> utl_helpers::Result<Self> {
        utl_helpers::config::load_or_create(&Self::config_path())
    }

    pub fn save(&self) -> utl_helpers::Result {
        utl_helpers::config::save(self, &Self::config_path())
    }

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
