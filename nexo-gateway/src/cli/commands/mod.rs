//! Command handlers for the `nexo-gateway` CLI.

/// Initialize gateway configuration and local storage.
pub mod init;

/// Generate protocol schemas.
pub mod schema;

/// Start the gateway runtime.
pub mod start;

use nexo_core::GatewayProperties;
use std::path::PathBuf;

pub(crate) fn gateway_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-gateway.toml")
}

pub(crate) fn save_gateway_properties(properties: &GatewayProperties) -> cli_helpers::Result {
    cli_helpers::config::save(properties, &gateway_config_path())
}
