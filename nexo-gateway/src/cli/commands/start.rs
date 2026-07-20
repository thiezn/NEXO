use crate::{NexoGateway, Result};
use std::path::PathBuf;

/// Command handler for the `start` command of the `nexo-gateway` CLI.
pub async fn run(host: Option<String>, port: Option<u16>) -> Result {
    let path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-gateway.toml");
    let mut config = if path.exists() {
        let config: nexo_core::GatewayProperties = cli_helpers::config::load(&path)?;
        config.into_builder().build()
    } else {
        let config = nexo_core::GatewayProperties::new(
            nexo_core::ClientInfo::new(env!("CARGO_PKG_VERSION")),
            nexo_ws_schema::AUTH_TOKEN,
        );
        cli_helpers::config::save(&config, &path)?;
        config
    };
    if let Some(h) = host {
        config = config.into_builder().host(h).build();
    }
    if let Some(p) = port {
        config = config.into_builder().port(p).build();
    }
    NexoGateway::new(config)?.run().await
}
