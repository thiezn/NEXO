use anyhow::{Result, bail};

use crate::config::{AppConfig, ImageModelPaths};
use crate::inference::pipelines::flux2::Flux2Engine;
use crate::inference::pipelines::qwen_image::QwenImageEngine;
use crate::inference::pipelines::zimage::ZImageEngine;
use crate::inference::{InferenceEngine, LoadStrategy};

/// Determine the model family from config or manifest, defaulting to "flux2".
fn resolve_family(model_name: &str, config: &AppConfig) -> String {
    let model_cfg = config.model_config(model_name);
    if let Some(family) = model_cfg.family {
        return family;
    }
    if let Some(manifest) = crate::manifest::find_manifest(model_name) {
        return manifest.family.clone();
    }
    "flux2".to_string()
}

fn resolve_qwen3_variant(config: &AppConfig) -> Option<String> {
    std::env::var("nexo_QWEN3_VARIANT")
        .ok()
        .or_else(|| config.qwen3_variant.clone())
}

/// Create an inference engine for the given model, auto-detecting the family.
pub fn create_engine(
    model_name: String,
    paths: ImageModelPaths,
    config: &AppConfig,
    load_strategy: LoadStrategy,
) -> Result<Box<dyn InferenceEngine>> {
    let family = resolve_family(&model_name, config);

    match family.as_str() {
        "flux2" | "flux.2" | "flux2-klein" => Ok(Box::new(Flux2Engine::new(
            model_name,
            paths,
            resolve_qwen3_variant(config),
            load_strategy,
        ))),
        "z-image" => Ok(Box::new(ZImageEngine::new(
            model_name,
            paths,
            resolve_qwen3_variant(config),
            load_strategy,
        ))),
        "qwen-image" | "qwen_image" => Ok(Box::new(QwenImageEngine::new(
            model_name,
            paths,
            load_strategy,
        ))),
        other => bail!(
            "unknown model family '{}' for model '{}'. Supported: flux2, z-image, qwen-image",
            other,
            model_name
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dummy_paths() -> ImageModelPaths {
        ImageModelPaths {
            transformer: PathBuf::from("/tmp/transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/vae"),
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
        }
    }

    #[test]
    fn resolve_family_from_manifest() {
        let config = AppConfig::default();
        assert_eq!(resolve_family("z-image-turbo:q8", &config), "z-image");
        assert_eq!(resolve_family("flux2-klein:bf16", &config), "flux2");
        assert_eq!(resolve_family("qwen-image:bf16", &config), "qwen-image");
    }

    #[test]
    fn resolve_family_unknown_defaults_to_flux2() {
        let config = AppConfig::default();
        assert_eq!(resolve_family("totally-unknown-model", &config), "flux2");
    }

    #[test]
    fn create_engine_flux2() {
        let config = AppConfig::default();
        let engine = create_engine(
            "flux2-klein:bf16".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "flux2-klein:bf16");
    }

    #[test]
    fn create_engine_qwen_image() {
        let mut config = AppConfig::default();
        config.models.insert(
            "my-qwen-image".to_string(),
            crate::config::ModelConfig {
                family: Some("qwen-image".to_string()),
                ..Default::default()
            },
        );
        let engine = create_engine(
            "my-qwen-image".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "my-qwen-image");
    }

    #[test]
    fn create_engine_zimage() {
        let config = AppConfig::default();
        let engine = create_engine(
            "z-image-turbo:q8".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        )
        .unwrap();
        assert_eq!(engine.model_name(), "z-image-turbo:q8");
    }

    #[test]
    fn create_engine_unknown_family_fails() {
        let mut config = AppConfig::default();
        config.models.insert(
            "bad-model".to_string(),
            crate::config::ModelConfig {
                family: Some("nosuchfamily".to_string()),
                ..Default::default()
            },
        );
        let result = create_engine(
            "bad-model".to_string(),
            dummy_paths(),
            &config,
            LoadStrategy::Sequential,
        );
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("nosuchfamily"));
    }
}
