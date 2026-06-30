/// Command handler for the `start` command of the `nexo-gateway` CLI.
pub async fn run(host: Option<String>, port: Option<u16>) -> cli_helpers::Result {
    let mut config = config::GatewayConfig::load()?;
    if let Some(h) = host {
        config.host = h;
    }
    if let Some(p) = port {
        config.port = p;
    }
    runtime::run(&config).await
}
