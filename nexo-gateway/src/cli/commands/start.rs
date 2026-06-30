use crate::{NexoGateway, Result};

/// Command handler for the `start` command of the `nexo-gateway` CLI.
pub async fn run(host: Option<String>, port: Option<u16>) -> Result {
    let path = super::gateway_config_path();
    let mut config = if path.exists() {
        let config: nexo_core::GatewayProperties = cli_helpers::config::load(&path)?;
        config.into_builder().build()
    } else {
        let config = nexo_core::GatewayProperties::new(
            nexo_core::ClientInfo::new(env!("CARGO_PKG_VERSION")),
            nexo_ws_schema::AUTH_TOKEN,
        );
        super::save_gateway_properties(&config)?;
        config
    };
    if let Some(h) = host {
        config = config.into_builder().host(h).build();
    }
    if let Some(p) = port {
        config = config.into_builder().port(p).build();
    }
    NexoGateway::new(config).run().await
}
