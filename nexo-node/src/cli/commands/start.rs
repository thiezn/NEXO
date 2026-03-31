use std::path::PathBuf;

use crate::config::NodeConfig;
use crate::download::{default_models_dir, find_manifest, known_manifests, storage_path};
use crate::download::registry::DEFAULT_INFERENCE_MODEL;
use crate::registry::ToolRegistry;
use crate::services::ServiceManager;

pub async fn run(url: Option<String>) -> utl_helpers::Result {
    let mut config = NodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    // Auto-detect which registered models are downloaded on disk.
    config.available_models = detect_available_models();
    if !config.available_models.is_empty() {
        tracing::info!("Detected available models: {:?}", config.available_models);
    }

    // Try to start inference services; warn but continue if unavailable.
    let services = try_start_services().await;
    let has_inference = services.is_some();

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

    crate::connect::run_node(&config, &registry, has_inference).await
    // services is dropped here → monitor task aborted → llama-server killed
}

/// Scan the models directory to find which registered models are downloaded.
fn detect_available_models() -> Vec<String> {
    let mdir = default_models_dir();
    known_manifests()
        .iter()
        .filter(|m| m.files.iter().all(|f| mdir.join(storage_path(m, f)).exists()))
        .map(|m| m.name.clone())
        .collect()
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
