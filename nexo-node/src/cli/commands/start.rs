use std::path::PathBuf;

use crate::config::NodeConfig;
use crate::download::{default_models_dir, find_manifest, storage_path};
use crate::download::registry::DEFAULT_INFERENCE_MODEL;
use crate::registry::ToolRegistry;
use crate::services::ServiceManager;

pub async fn run(url: Option<String>) -> utl_helpers::Result {
    let mut config = NodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    // Try to start inference services; warn but continue if unavailable.
    let _services = try_start_services().await;

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

    crate::connect::run_node(&config, &registry).await
    // _services is dropped here → monitor task aborted → llama-server killed
}

async fn try_start_services() -> Option<ServiceManager> {
    let model_path = resolve_model_path(DEFAULT_INFERENCE_MODEL)?;
    match ServiceManager::start(model_path).await {
        Ok(s) => {
            tracing::info!("llama-server started successfully");
            Some(s)
        }
        Err(e) => {
            eprintln!("warning: could not start inference service: {e}");
            None
        }
    }
}

fn resolve_model_path(model_name: &str) -> Option<PathBuf> {
    let manifest = find_manifest(model_name)?;
    let file = manifest.files.first()?;
    let path = default_models_dir().join(storage_path(manifest, file));

    if !path.exists() {
        eprintln!(
            "warning: model not found at {}\n  Run: nexo-node models pull {model_name}",
            path.display()
        );
        return None;
    }

    Some(path)
}
