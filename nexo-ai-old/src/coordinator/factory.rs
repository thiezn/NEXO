use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::api::model_traits::ModelInfo;
use crate::api::types::ModelCategory;
use crate::config::CoordinatorConfig;
#[cfg(feature = "candle")]
use crate::registry::CandleBackend;
#[cfg(feature = "mlx")]
use crate::registry::OpenAiProvider;
use crate::registry::{AiModelManifest, ModelFamily, ModelRuntime};
#[cfg(feature = "mlx")]
use crate::inference::remote::servers::ManagedProviderServers;

pub(super) struct ModelFactory<'a> {
    config: &'a CoordinatorConfig,
    #[cfg(feature = "mlx")]
    provider_servers: &'a mut ManagedProviderServers,
}

impl<'a> ModelFactory<'a> {
    #[cfg(feature = "mlx")]
    pub(super) fn new(
        config: &'a CoordinatorConfig,
        provider_servers: &'a mut ManagedProviderServers,
    ) -> Self {
        Self {
            config,
            provider_servers,
        }
    }

    #[cfg(not(feature = "mlx"))]
    pub(super) fn new(config: &'a CoordinatorConfig) -> Self {
        Self { config }
    }

    pub(super) fn build(
        &mut self,
        manifest: &AiModelManifest,
        model_name: &str,
        model_dir: PathBuf,
        memory_bytes: u64,
        categories: Vec<ModelCategory>,
    ) -> Result<Box<dyn ModelInfo>> {
        #[cfg(not(feature = "mlx"))]
        let _ = &categories;

        match (manifest.family, &manifest.runtime) {
            #[cfg(feature = "candle")]
            (ModelFamily::Whisper, ModelRuntime::Candle(CandleBackend::Safetensors)) => {
                Ok(Box::new(crate::inference::models::whisper::WhisperModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )))
            }
            #[cfg(feature = "candle")]
            (ModelFamily::Flux, ModelRuntime::Candle(CandleBackend::Safetensors)) => {
                Ok(Box::new(crate::inference::models::flux2::FluxModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )))
            }
            #[cfg(feature = "candle")]
            (ModelFamily::ZImage, ModelRuntime::Candle(CandleBackend::Gguf)) => {
                Ok(Box::new(crate::inference::models::z_image::ZImageModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )))
            }
            #[cfg(feature = "candle")]
            (ModelFamily::QwenImage, ModelRuntime::Candle(_)) => {
                Ok(Box::new(crate::inference::models::qwen_image::QwenImageModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )))
            }
            #[cfg(feature = "candle")]
            (ModelFamily::Gemma4, ModelRuntime::Candle(backend)) => {
                let settings = self.config.model_settings(model_name);
                Ok(Box::new(
                    crate::inference::models::gemma4::Gemma4Model::new(
                        model_name.to_string(),
                        memory_bytes,
                        model_dir,
                    )
                    .with_gguf(matches!(backend, CandleBackend::Gguf))
                    .with_max_context_tokens(settings.max_context_tokens),
                ))
            }
            #[cfg(feature = "mlx")]
            (
                ModelFamily::Gemma4,
                ModelRuntime::OpenAi {
                    provider: OpenAiProvider::MlxVlm,
                    ..
                },
            ) => {
                let (host, port) = self.config.mlx_vlm_server_addr();
                let server = self.provider_servers.mlx_vlm(self.config);
                let base_url = format!("http://{host}:{port}");
                Ok(Box::new(crate::inference::remote::openai::model::OpenAiModel::new(
                    model_name,
                    model_dir,
                    memory_bytes,
                    categories,
                    crate::inference::models::gemma4::openai::Gemma4OpenAiFamily,
                    server,
                    &base_url,
                )))
            }
            #[cfg(feature = "mlx")]
            (
                ModelFamily::Whisper,
                ModelRuntime::OpenAi {
                    provider: OpenAiProvider::MlxAudio,
                    ..
                },
            ) => {
                let (host, port) = self.config.mlx_audio_server_addr();
                let server = self.provider_servers.mlx_audio(self.config);
                let base_url = format!("http://{host}:{port}");
                Ok(Box::new(
                    crate::inference::models::whisper::openai::build_whisper_openai_model(
                        model_name.to_string(),
                        model_dir.to_string_lossy().into_owned(),
                        memory_bytes,
                        server,
                        &base_url,
                    ),
                ))
            }
            #[cfg(feature = "mlx")]
            (
                ModelFamily::Voxtral,
                ModelRuntime::OpenAi {
                    provider: OpenAiProvider::MlxAudio,
                    ..
                },
            ) => {
                let (host, port) = self.config.mlx_audio_server_addr();
                let server = self.provider_servers.mlx_audio(self.config);
                let base_url = format!("http://{host}:{port}");
                Ok(Box::new(
                    crate::inference::models::voxtral::openai::build_voxtral_openai_model(
                        model_name.to_string(),
                        model_dir.to_string_lossy().into_owned(),
                        memory_bytes,
                        server,
                        &base_url,
                    ),
                ))
            }
            _ => bail!(
                "unsupported model runtime combination: family={} backend={}",
                manifest.family,
                manifest.runtime.as_str()
            ),
        }
    }
}
