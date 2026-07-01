use nexo_ai::{InferenceEngine, ModelCatalog};
use nexo_core::{ClientInfo, DeviceInfo, NodeProperties, ToolRegistry};
use nexo_echo::EchoTool;
use nexo_node::{NexoNode, Result};
use std::sync::Arc;
use tracing::info;

use super::{node_config_path, save_node_properties};

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
    let path = node_config_path();
    let mut config = if path.exists() {
        let config: NodeProperties = cli_helpers::config::load(&path)?;
        config.into_builder().build()
    } else {
        let config = NodeProperties::new(
            ClientInfo::new(env!("CARGO_PKG_VERSION")),
            DeviceInfo::default(),
            nexo_ws_schema::AUTH_TOKEN,
        );
        save_node_properties(&config)?;
        config
    };
    if let Some(u) = url {
        config = config.into_builder().gateway_url(u).build();
    }

    info!("Loaded node configuration from {}", path.display());
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool)?;
    info!("Registered {} tool(s)", registry.len());

    let catalog = ModelCatalog::new();
    let local_available_manifests = catalog.list_downloaded_manifests(false);
    let engine = InferenceEngine::new(local_available_manifests)?;

    config = config
        .into_builder()
        .tools(registry.definitions())
        .models(engine.model_ids().into_iter().cloned().collect())
        .build();

    info!(
        "Starting nexo-node '{}' v{}",
        config.client().id,
        config.client().version
    );

    let engine = NexoNode::new(config, Arc::new(registry), Arc::new(engine));

    engine.run().await
}
