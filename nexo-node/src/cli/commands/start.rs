use std::path::PathBuf;

use crate::config::NodeConfig;
use crate::download::manifest::GgufComponent;
use crate::download::registry::DEFAULT_INFERENCE_MODEL;
use crate::download::{default_models_dir, find_manifest, known_manifests, storage_path};
use crate::registry::ToolRegistry;
use crate::services::ServiceManager;

struct ResolvedModel {
    weights_path: PathBuf,
    mmproj_path: Option<PathBuf>,
}

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
    let (services, has_vision) = match try_start_services().await {
        Some((s, vision)) => (Some(s), vision),
        None => (None, false),
    };
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

    crate::connect::run_node(&config, &registry, has_inference, has_vision).await
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

async fn try_start_services() -> Option<(ServiceManager, bool)> {
    let resolved = resolve_model_paths(DEFAULT_INFERENCE_MODEL)?;
    let has_vision = resolved.mmproj_path.is_some();
    match ServiceManager::start(resolved.weights_path, resolved.mmproj_path).await {
        Ok(s) => {
            tracing::info!(
                "llama-server started successfully{}",
                if has_vision { " (vision enabled)" } else { "" }
            );
            Some((s, has_vision))
        }
        Err(e) => {
            eprintln!("warning: could not start inference service: {e}");
            None
        }
    }
}

fn resolve_model_paths(model_name: &str) -> Option<ResolvedModel> {
    let manifest = find_manifest(model_name)?;
    let mdir = default_models_dir();

    let weights_file = manifest
        .files
        .iter()
        .find(|f| matches!(f.component, GgufComponent::Weights))?;
    let weights_path = mdir.join(storage_path(manifest, weights_file));

    if !weights_path.exists() {
        eprintln!(
            "warning: model not found at {}\n  Run: nexo-node models pull {model_name}",
            weights_path.display()
        );
        return None;
    }

    let mmproj_path = manifest
        .files
        .iter()
        .find(|f| matches!(f.component, GgufComponent::VisionProjector))
        .map(|f| mdir.join(storage_path(manifest, f)))
        .filter(|p| p.exists());

    Some(ResolvedModel {
        weights_path,
        mmproj_path,
    })
}
