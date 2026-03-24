use nexo_ws_schema::Platform;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Client configuration, stored at `~/.nexo/client.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClientConfig {
    pub gateway_url: String,
    pub client_id: String,
    pub client_version: String,
    pub platform: Platform,
    pub device_id: String,
    pub auth_token: String,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            gateway_url: "ws://127.0.0.1:6969".to_string(),
            client_id: "cli".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            platform: platform_from_os(),
            device_id: "default_device".to_string(),
            auth_token: nexo_ws_schema::AUTH_TOKEN.to_string(),
        }
    }
}

fn platform_from_os() -> Platform {
    match std::env::consts::OS {
        "macos" => Platform::Macos,
        "ios" => Platform::Ios,
        "linux" => Platform::Linux,
        "windows" => Platform::Windows,
        _ => Platform::Macos,
    }
}

impl ClientConfig {
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nexo")
            .join("client.toml")
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
        let config = ClientConfig::default();
        assert_eq!(config.gateway_url, "ws://127.0.0.1:6969");
        assert_eq!(config.client_id, "cli");
        assert_eq!(config.auth_token, nexo_ws_schema::AUTH_TOKEN);
    }

    #[test]
    fn config_roundtrip() {
        let config = ClientConfig {
            gateway_url: "ws://10.0.0.1:8080".into(),
            client_id: "test-client".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: ClientConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.gateway_url, "ws://10.0.0.1:8080");
        assert_eq!(decoded.client_id, "test-client");
    }

    #[test]
    fn default_platform_is_valid() {
        let config = ClientConfig::default();
        // Platform should serialize to a known lowercase string
        let json = serde_json::to_string(&config.platform).unwrap();
        assert!(
            json == "\"macos\"" || json == "\"ios\"" || json == "\"linux\"" || json == "\"windows\""
        );
    }
}
