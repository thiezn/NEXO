use crate::config::NodeConfig;
use crate::registry::ToolRegistry;
use nexo_ai::config::AiConfig;
use nexo_ai::coordinator::Coordinator;
use nexo_ai::download::manifest::storage_path;
use nexo_ai::download::paths::default_models_dir;
use nexo_ai::registry::known_manifests;
use std::sync::{Arc, Mutex};

pub async fn run(url: Option<String>) -> utl_helpers::Result {
    let mut config = NodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    // Build coordinator for local inference via nexo-ai.
    let ai_config = AiConfig::load().unwrap_or_default();
    let coordinator = Coordinator::new(ai_config);

    // Detect which nexo-ai models are downloaded on disk.
    let available_models = detect_available_models();
    if !available_models.is_empty() {
        tracing::info!("Detected available models: {:?}", available_models);
    }

    tracing::info!(
        "Starting nexo-node '{}' v{}",
        config.node_id,
        config.node_version
    );

    let registry = ToolRegistry::with_builtins();
    tracing::info!(
        "Loaded {} tool(s): {:?}",
        registry.tool_count(),
        registry.specs().iter().map(|s| &s.name).collect::<Vec<_>>()
    );

    let coordinator = Arc::new(Mutex::new(coordinator));
    crate::connect::run_node(&config, &available_models, &registry, coordinator).await
}

/// Scan the nexo-ai models directory to find which registered models are downloaded.
fn detect_available_models() -> Vec<String> {
    let mdir = default_models_dir();
    known_manifests()
        .iter()
        .filter(|m| {
            m.manifest
                .files
                .iter()
                .all(|f| mdir.join(storage_path(&m.manifest, f)).exists())
        })
        .map(|m| m.manifest.name.clone())
        .collect()
}
