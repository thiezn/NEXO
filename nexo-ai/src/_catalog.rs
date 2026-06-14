//! Catalog adapters from NEXO model manifests into `nexo-ai` runtime configs.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nexo_model_mgmt::registry::{known_manifests, list_models};
use nexo_model_mgmt::{
    AnyTtsManifestEngine, ManifestModelDataType, ManifestRuntimeBinding, MistralRsManifestLoader,
    ModelManifest, MoldManifestLoader,
};

use crate::ModelRuntimeImplementation;
use crate::engine::any_tts::AnyTtsModelConfig;
use crate::engine::mistralrs::{
    MistralRsAutoLoader, MistralRsDiffusionLoader, MistralRsGgufLoader, MistralRsLoader,
    MistralRsModelConfig, MistralRsSpeechLoader,
};
use crate::engine::mold::{MoldFlux2Loader, MoldLoader, MoldModelConfig};
use crate::{Error, ModelDataType, RegisteredModelConfig, Result};

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
/// * `manifest` - The manifest describing local model identity, runtime bindings, and files.
///
/// # Errors
///
/// Returns an error when a manifest runtime binding does not contain enough loader data.
pub fn model_config_from_manifest(manifest: &ModelManifest) -> Result<RegisteredModelConfig> {
    Ok(RegisteredModelConfig {
        descriptor: manifest.descriptor.clone(),
        runtimes: manifest
            .runtime_bindings
            .iter()
            .map(|binding| runtime_config_from_manifest_binding(manifest, binding))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn runtime_config_from_manifest_binding(
    manifest: &ModelManifest,
    binding: &ManifestRuntimeBinding,
) -> Result<ModelRuntimeImplementation> {
    match binding {
        ManifestRuntimeBinding::AnyTts(binding) => match binding.engine {
            AnyTtsManifestEngine::Kokoro => Ok(ModelRuntimeImplementation::AnyTts(
                AnyTtsModelConfig::Kokoro,
            )),
        },
        ManifestRuntimeBinding::MistralRs(binding) => Ok(ModelRuntimeImplementation::MistralRs(
            MistralRsModelConfig {
                loader: mistral_loader_from_manifest_binding(manifest, &binding.loader)?,
                revision: binding.revision.clone(),
            },
        )),
        ManifestRuntimeBinding::Mold(binding) => match binding.loader {
            MoldManifestLoader::Flux2 => Ok(ModelRuntimeImplementation::Mold(MoldModelConfig {
                loader: MoldLoader::Flux2(MoldFlux2Loader {
                    model_id: manifest.id().to_string(),
                }),
            })),
        },
    }
}

fn mistral_loader_from_manifest_binding(
    manifest: &ModelManifest,
    loader: &MistralRsManifestLoader,
) -> Result<MistralRsLoader> {
    match loader {
        MistralRsManifestLoader::Auto(loader) => Ok(MistralRsLoader::Auto(MistralRsAutoLoader {
            model_id: manifest.id().to_string(),
            from_uqff: loader
                .from_uqff
                .as_ref()
                .map(|paths| paths.iter().map(PathBuf::from).collect()),
            tokenizer_json: None,
            chat_template: None,
            jinja_explicit: None,
            dtype: model_data_type(loader.dtype),
            hf_cache_path: None,
        })),
        MistralRsManifestLoader::Gguf(loader) => {
            if loader.quantized_filenames.is_empty() {
                return Err(Error::Config {
                    message: format!(
                        "GGUF model manifest '{}' does not declare quantized filenames",
                        manifest.id()
                    ),
                });
            }

            Ok(MistralRsLoader::Gguf(MistralRsGgufLoader {
                tokenizer_model_id: None,
                quantized_model_id: manifest.id().to_string(),
                quantized_filenames: loader.quantized_filenames.clone(),
                chat_template: None,
                jinja_explicit: None,
                dtype: model_data_type(loader.dtype),
            }))
        }
        MistralRsManifestLoader::Diffusion(loader) => {
            Ok(MistralRsLoader::Diffusion(MistralRsDiffusionLoader {
                model_id: manifest.id().to_string(),
                offload: loader.offload,
                dtype: model_data_type(loader.dtype),
            }))
        }
        MistralRsManifestLoader::Speech(loader) => {
            let dac_model_id = loader.dac_subdir.as_ref().map(|subdir| {
                nexo_model_mgmt::resolve_model_storage_dir(manifest.id())
                    .join(subdir)
                    .to_string_lossy()
                    .into_owned()
            });

            Ok(MistralRsLoader::Speech(MistralRsSpeechLoader {
                model_id: manifest.id().to_string(),
                dac_model_id,
                dtype: model_data_type(loader.dtype),
            }))
        }
    }
}

const fn model_data_type(dtype: ManifestModelDataType) -> ModelDataType {
    match dtype {
        ManifestModelDataType::Auto => ModelDataType::Auto,
        ManifestModelDataType::Bf16 => ModelDataType::Bf16,
        ManifestModelDataType::F16 => ModelDataType::F16,
        ManifestModelDataType::F32 => ModelDataType::F32,
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

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Gguf(loader),
            ..
        }) = &config.runtimes[0]
        else {
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

        assert!(matches!(
            &config.runtimes[0],
            ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
                loader: MistralRsLoader::Auto(_),
                ..
            })
        ));
    }

    #[test]
    fn gemma_4_uqff_manifest_uses_auto_dtype() {
        let manifest = find_manifest("gemma-4-e4b-it-uqff-afq8").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Auto(loader),
            ..
        }) = &config.runtimes[0]
        else {
            panic!("expected auto loader");
        };

        assert_eq!(loader.dtype, ModelDataType::Auto);
        assert_eq!(loader.from_uqff, Some(vec![PathBuf::from("afq8-")]));
    }

    #[test]
    fn gemma_4_12b_safetensors_manifest_uses_auto_loader() {
        let manifest = find_manifest("gemma-4-12b-it").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Auto(loader),
            ..
        }) = &config.runtimes[0]
        else {
            panic!("expected auto loader");
        };

        assert_eq!(loader.model_id, "gemma-4-12b-it");
        assert_eq!(loader.dtype, ModelDataType::Auto);
        assert_eq!(loader.from_uqff, None);
    }

    #[test]
    fn gemma_4_12b_uqff_manifest_uses_auto_loader() {
        let manifest = find_manifest("gemma-4-12b-it-uqff-q4k").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Auto(loader),
            ..
        }) = &config.runtimes[0]
        else {
            panic!("expected auto loader");
        };

        assert_eq!(loader.dtype, ModelDataType::Auto);
        assert_eq!(loader.from_uqff, Some(vec![PathBuf::from("q4k-")]));
    }

    #[test]
    fn flux_manifest_uses_diffusion_loader() {
        let manifest = find_manifest("flux.2-klein-9b").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Diffusion(loader),
            ..
        }) = &config.runtimes[0]
        else {
            panic!("expected diffusion loader");
        };

        assert_eq!(loader.model_id, "flux.2-klein-9b");
        assert_eq!(loader.dtype, ModelDataType::Auto);
    }

    #[test]
    fn flux_manifest_also_exposes_mold_runtime() {
        let manifest = find_manifest("flux.2-klein-9b").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let mold = config
            .runtimes
            .iter()
            .find_map(|runtime| match runtime {
                ModelRuntimeImplementation::Mold(model) => Some(model),
                _ => None,
            })
            .expect("expected mold runtime for flux.2 manifest");

        let MoldLoader::Flux2(loader) = &mold.loader;

        assert_eq!(loader.model_id, "flux.2-klein-9b");
    }

    #[test]
    fn dia_manifest_uses_speech_loader() {
        let manifest = find_manifest("dia-1.6b-tts").unwrap();
        let config = model_config_from_manifest(manifest).unwrap();

        let ModelRuntimeImplementation::MistralRs(MistralRsModelConfig {
            loader: MistralRsLoader::Speech(loader),
            ..
        }) = &config.runtimes[0]
        else {
            panic!("expected speech loader");
        };

        assert_eq!(loader.model_id, "dia-1.6b-tts");
        assert!(
            loader
                .dac_model_id
                .as_deref()
                .is_some_and(|path| path.ends_with("dia-1.6b/dac"))
        );
        assert_eq!(loader.dtype, ModelDataType::F16);
    }
}
