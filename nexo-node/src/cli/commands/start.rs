use nexo_node::{NexoNode, NexoNodeConfig, Result};

/// Start the node runtime, auto-load configured models, and connect to the gateway.
///
/// # Arguments
///
/// * `url` - Optional gateway URL override for this launch.
///
/// # Errors
///
/// Returns an error if configuration loading, model startup, or the node runtime fails.
pub async fn run(url: Option<String>) -> Result {
    let mut config = NexoNodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    tracing::info!(
        "Starting nexo-node '{}' v{}",
        config.node_id,
        config.node_version
    );

    let engine = NexoNode::new(config)?;

    engine.run().await?;

    Ok(())
}
