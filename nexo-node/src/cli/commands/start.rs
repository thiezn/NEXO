use nexo_node::{NexoEngine, NexoNodeConfig};
/// Start the node runtime, auto-load configured models, and connect to the gateway.
///
/// # Arguments
///
/// * `url` - Optional gateway URL override for this launch.
///
/// # Errors
///
/// Returns an error if configuration loading, model startup, or the node runtime fails.
pub async fn run(url: Option<String>) -> cli_helpers::Result {
    let mut config = NexoNodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    // Detect which catalog models are downloaded on disk and can be exposed through nexo-ai.
    tracing::info!("Detecting available models on disk...");
    let available_models = nexo_ai::downloaded_model_configs()
        .map_err(|error| cli_helpers::Error::Other(error.to_string()))?;

    tracing::info!(
        "Starting nexo-node '{}' v{}",
        config.node_id,
        config.node_version
    );

    let registry = nexo_echo::tool_registry().map_err(|error| {
        cli_helpers::Error::Other(format!("failed to initialize node tools: {error}"))
    })?;
    tracing::info!(
        "Loaded {} tool(s): {:?}",
        registry.len(),
        registry
            .definitions()
            .iter()
            .map(|definition| &definition.name)
            .collect::<Vec<_>>()
    );

    nexo_node::transport::run_node(&config, &registry, available_models).await
}
