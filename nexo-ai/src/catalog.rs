//! Catalog adapters from NEXO model manifests into `nexo-ai` runtime configs.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nexo_model_mgmt::registry::{known_manifests, list_models};
use nexo_model_mgmt::{ModelComponent, ModelFileSelector, ModelManifest};

use crate::{
    AutoModelLoader, Error, GgufModelLoader, ModelDataType, ModelLoader, RegisteredModelConfig,
    Result,
};

/// Build runtime configs for every downloaded model known to `nexo-model-mgmt`.
///
/// # Errors
///
/// Returns an error if a downloaded manifest cannot be represented by the `nexo-ai` runtime.
pub fn downloaded_model_configs() -> Result<Vec<RegisteredModelConfig>> {
    let downloaded = list_models()
        .into_iter()
        .filter(|entry| entry.is_downloaded)
        .map(|entry| entry.id)
        .collect::<BTreeSet<_>>();

    known_manifests()
        .iter()
        .filter(|manifest| downloaded.contains(manifest.id()))
        .map(model_config_from_manifest)
        .collect()
}

/// Convert a shared model manifest into a `nexo-ai` runtime model config.
///
/// # Arguments
///
/// * `manifest` - The manifest describing local model identity, backend, and files.
///
/// # Errors
///
/// Returns an error when the manifest does not contain enough information for the selected backend.
pub fn model_config_from_manifest(manifest: &ModelManifest) -> Result<RegisteredModelConfig> {
    Ok(RegisteredModelConfig {
        descriptor: manifest.descriptor.clone(),
        loader: loader_from_manifest(manifest)?,
        revision: None,
    })
}

fn loader_from_manifest(manifest: &ModelManifest) -> Result<ModelLoader> {
    if manifest.backend == "mistralrs-gguf" {
        return Ok(ModelLoader::Gguf(GgufModelLoader {
            tokenizer_model_id: None,
            quantized_model_id: manifest.id().to_string(),
            quantized_filenames: gguf_filenames(manifest)?,
            chat_template: None,
            jinja_explicit: None,
            dtype: ModelDataType::Auto,
        }));
    }

    Ok(ModelLoader::Auto(AutoModelLoader {
        model_id: manifest.id().to_string(),
        from_uqff: uqff_selectors(manifest),
        tokenizer_json: None,
        chat_template: None,
        jinja_explicit: None,
        dtype: ModelDataType::Auto,
        hf_cache_path: None,
    }))
}

fn uqff_selectors(manifest: &ModelManifest) -> Option<Vec<PathBuf>> {
    let selectors = manifest
        .files
        .iter()
        .filter(|file| file.component == ModelComponent::UqffShard)
        .filter_map(|file| match &file.selector {
            ModelFileSelector::Exact(path) | ModelFileSelector::Prefix(path) => {
                Some(PathBuf::from(path))
            }
            ModelFileSelector::Suffix(_) => None,
        })
        .collect::<Vec<_>>();

    (!selectors.is_empty()).then_some(selectors)
}

fn gguf_filenames(manifest: &ModelManifest) -> Result<Vec<String>> {
    let filenames = manifest
        .files
        .iter()
        .filter(|file| {
            matches!(
                file.component,
                ModelComponent::Weights | ModelComponent::VisionProjector
            )
        })
        .filter_map(|file| match &file.selector {
            ModelFileSelector::Exact(path) if path.ends_with(".gguf") => Some(path.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if filenames.is_empty() {
        Err(Error::Config {
            message: format!(
                "GGUF model manifest '{}' does not contain exact .gguf filenames",
                manifest.id()
            ),
        })
    } else {
        Ok(filenames)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used)]

    use nexo_model_mgmt::registry::find_manifest;

    use super::*;

    #[test]
    fn gguf_manifest_uses_explicit_gguf_filenames() {
        let manifest = find_manifest("gemma-4-e2b-it-q5").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelLoader::Gguf(loader) = config.loader else {
            panic!("expected gguf loader");
        };

        assert!(
            loader
                .quantized_filenames
                .iter()
                .any(|name| name.ends_with(".gguf"))
        );
    }

    #[test]
    fn uqff_manifest_uses_auto_loader() {
        let manifest = find_manifest("gemma-4-e2b-it-uqff-q4k").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        assert!(matches!(config.loader, ModelLoader::Auto(_)));
    }

    #[test]
    fn gemma_4_uqff_manifest_uses_auto_dtype() {
        let manifest = find_manifest("gemma-4-e4b-it-uqff-afq8").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelLoader::Auto(loader) = config.loader else {
            panic!("expected auto loader");
        };

        assert_eq!(loader.dtype, ModelDataType::Auto);
        assert_eq!(loader.from_uqff, Some(vec![PathBuf::from("afq8-")]));
    }
}
