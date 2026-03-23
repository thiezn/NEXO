use anyhow::Result;

use crate::config::{AppConfig, TTSModelPaths};
use crate::manifest::find_manifest;

use super::pipelines::parler::ParlerEngine;
use super::pipelines::qwen3::Qwen3TTSEngine;
use super::{InferenceEngine, LoadStrategy};

/// Create the appropriate inference engine for the given model.
pub fn create_engine(
    model_name: &str,
    paths: TTSModelPaths,
    config: &AppConfig,
    load_strategy: LoadStrategy,
) -> Result<Box<dyn InferenceEngine>> {
    let family = resolve_family(model_name, config);

    match family.as_str() {
        "parler" => Ok(Box::new(ParlerEngine::new(
            model_name.to_string(),
            paths,
            load_strategy,
        ))),
        "qwen3" => Ok(Box::new(Qwen3TTSEngine::new(
            model_name.to_string(),
            paths,
            load_strategy,
        ))),
        _ => anyhow::bail!("Unknown model family '{family}' for model '{model_name}'"),
    }
}

/// Determine the model family from config or manifest.
fn resolve_family(model_name: &str, config: &AppConfig) -> String {
    // Check config first
    if let Some(model_cfg) = config.models.get(model_name)
        && let Some(ref family) = model_cfg.family
    {
        return family.clone();
    }

    // Fall back to manifest
    if let Some(manifest) = find_manifest(model_name) {
        return manifest.family.clone();
    }

    // Default heuristic
    if model_name.starts_with("parler") {
        "parler".to_string()
    } else if model_name.starts_with("qwen3") {
        "qwen3".to_string()
    } else {
        "unknown".to_string()
    }
}
