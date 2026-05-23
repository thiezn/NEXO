use crate::config::NodeConfig;
use crate::tools::ToolRegistry;
use nexo_ai::api::types::ModelCategory;
use nexo_ai::coordinator::Coordinator;
use nexo_ai::registry::{detect_available_models, find_manifest, manifests_for_category};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

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
    let mut config = NodeConfig::load()?;
    if let Some(u) = url {
        config.gateway_url = u;
    }

    // Detect which nexo-ai models are downloaded on disk (with SHA-256 verification).
    tracing::info!("Detecting available models on disk...");
    let available_models = detect_available_models();
    if available_models.is_empty() {
        tracing::info!("No downloaded models detected");
    } else {
        tracing::info!("Detected available models: {:?}", available_models);
    }

    // Build coordinator from node config (not nexo-ai.toml).
    let mut coordinator = Coordinator::new(config.to_coordinator_config());

    // Smart startup: load minimum set of models to cover requested categories.
    let startup_models = resolve_startup_models(&config, &available_models);
    tracing::info!(
        "Startup categories: {:?}, models to load: {:?}",
        config.startup_categories,
        startup_models
    );
    for model_name in &startup_models {
        match coordinator.load_model(model_name) {
            Ok(()) => {
                if let Some(manifest) = find_manifest(model_name) {
                    for cat in &manifest.categories {
                        coordinator.set_active_model(*cat, model_name.clone());
                    }
                }
                tracing::info!("Auto-loaded startup model '{model_name}'");
            }
            Err(e) => tracing::warn!("Failed to auto-load '{model_name}': {e}"),
        }
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
    crate::transport::run_node(&config, &available_models, &registry, coordinator).await
}

/// Determine the minimum set of models to load at startup, deduplicating
/// across categories (one multi-category model can satisfy several categories).
fn resolve_startup_models(config: &NodeConfig, available_models: &[String]) -> Vec<String> {
    let categories: Vec<ModelCategory> = config
        .startup_categories
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    if categories.is_empty() {
        return vec![];
    }

    let mut needed: Vec<String> = Vec::new();
    let mut satisfied: HashSet<ModelCategory> = HashSet::new();

    for cat in &categories {
        if satisfied.contains(cat) {
            tracing::debug!("Category '{cat}' already satisfied by a loaded model");
            continue;
        }

        let model_name = if let Some(configured) = config.default_models.get(cat.as_str()) {
            configured.clone()
        } else {
            match auto_select_model(*cat, available_models) {
                Some(name) => name,
                None => {
                    tracing::warn!("No available model for startup category '{cat}'");
                    continue;
                }
            }
        };

        if !available_models.contains(&model_name) {
            tracing::warn!(
                "Model '{model_name}' for category '{cat}' is not downloaded — skipping"
            );
            continue;
        }

        // Mark all categories this model covers as satisfied.
        if let Some(manifest) = find_manifest(&model_name) {
            for supported_cat in &manifest.categories {
                satisfied.insert(*supported_cat);
            }
        }

        if !needed.contains(&model_name) {
            needed.push(model_name);
        }
    }

    needed
}

/// Auto-select the smallest available model that supports the given category.
fn auto_select_model(category: ModelCategory, available_models: &[String]) -> Option<String> {
    let mut candidates: Vec<_> = manifests_for_category(category)
        .into_iter()
        .filter(|m| available_models.contains(&m.manifest.name))
        .collect();

    candidates.sort_by(|a, b| {
        a.manifest
            .size_gb
            .partial_cmp(&b.manifest.size_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.first().map(|m| m.manifest.name.clone())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn resolve_empty_categories_returns_empty() {
        let config = NodeConfig {
            startup_categories: vec![],
            ..Default::default()
        };
        assert!(resolve_startup_models(&config, &[]).is_empty());
    }

    #[test]
    fn resolve_deduplicates_multi_category_model() {
        // gemma-4-e2b-it supports Chat + Tool + Image
        let available = vec!["gemma-4-e2b-it".to_string()];
        let config = NodeConfig {
            startup_categories: vec!["chat".into(), "tool".into(), "image".into()],
            ..Default::default()
        };
        let result = resolve_startup_models(&config, &available);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "gemma-4-e2b-it");
    }

    #[test]
    fn resolve_uses_configured_default() {
        let available = vec!["gemma-4-e2b-it".to_string(), "gemma-4-e4b-it".to_string()];
        let mut default_models = std::collections::HashMap::new();
        default_models.insert("chat".into(), "gemma-4-e4b-it".into());

        let config = NodeConfig {
            startup_categories: vec!["chat".into()],
            default_models,
            ..Default::default()
        };
        let result = resolve_startup_models(&config, &available);
        assert_eq!(result, vec!["gemma-4-e4b-it"]);
    }

    #[test]
    fn resolve_skips_unavailable_model() {
        let config = NodeConfig {
            startup_categories: vec!["chat".into()],
            ..Default::default()
        };
        // No models available on disk
        let result = resolve_startup_models(&config, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_different_families_returns_multiple() {
        // chat needs gemma4, listen needs whisper
        let available = vec!["gemma-4-e2b-it".to_string(), "distil-large-v3".to_string()];
        let config = NodeConfig {
            startup_categories: vec!["chat".into(), "listen".into()],
            ..Default::default()
        };
        let result = resolve_startup_models(&config, &available);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn resolve_prefers_smallest_available_mlx_model() {
        let available = vec![
            "gemma-4-e2b-it".to_string(),
            "mlx-gemma-4-e2b-it-8bit".to_string(),
        ];
        let config = NodeConfig {
            startup_categories: vec!["chat".into(), "tool".into(), "image".into()],
            ..Default::default()
        };

        let result = resolve_startup_models(&config, &available);

        assert_eq!(result, vec!["mlx-gemma-4-e2b-it-8bit".to_string()]);
    }
}
