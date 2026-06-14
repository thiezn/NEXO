use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, stream};
use mold_ai_core::{GenerateRequest as MoldGenerateRequest, ModelPaths as MoldModelPaths};
use mold_ai_inference::{
    Flux2Engine, InferenceEngine as MoldEngine, LoadStrategy as MoldLoadStrategy,
};
use nexo_core::inference::request::{GeneratedImage, ImageGenerationRequest};
use nexo_core::{
    InferenceErrorCode, InferenceFailure, InferenceRequest, InferenceResponse, InferenceRuntime,
    InferenceStream, MediaSource, ModelDefinition, ModelId, RequestId, Retryability,
};
use nexo_model_mgmt::resolve_model_storage_dir;

use super::{
    MoldFlux2Loader, MoldLoadStrategy as ConfigLoadStrategy, MoldLoader, MoldModelConfig,
    MoldRuntimeConfig,
};
use crate::{Error, RegisteredModelConfig, Result, RuntimeConfig};

const DEFAULT_FLUX2_STEPS: u32 = 4;

type SharedMoldEngine = Arc<Mutex<Box<dyn MoldEngine>>>;

#[derive(Clone)]
pub(crate) struct MoldRuntime {
    engine: SharedMoldEngine,
}

impl std::fmt::Debug for MoldRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoldRuntime").finish_non_exhaustive()
    }
}

impl MoldRuntime {
    pub(crate) async fn from_model_config(
        base_runtime_config: &RuntimeConfig,
        model: &RegisteredModelConfig,
    ) -> Result<Self> {
        let runtime_config = mold_runtime_config(base_runtime_config);
        let model_config = mold_model_config(model)?.clone();
        let model_name = model.descriptor.id.to_string();

        tokio::task::spawn_blocking(move || {
            let mut engine = build_engine(&runtime_config, &model_name, &model_config)?;
            engine.load().map_err(map_mold_error)?;
            Ok(Self {
                engine: Arc::new(Mutex::new(engine)),
            })
        })
        .await
        .map_err(|error| Error::Runtime {
            message: format!("failed to join mold model-load task: {error}"),
        })?
    }

    pub(crate) async fn submit(
        &self,
        descriptor: ModelDefinition,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match request {
            InferenceRequest::GenerateImage(request) => {
                self.submit_generate_image(descriptor, request).await
            }
            other => Err(Error::UnsupportedRequest {
                kind: request_kind(&other),
            }),
        }
    }

    async fn submit_generate_image(
        &self,
        descriptor: ModelDefinition,
        request: ImageGenerationRequest,
    ) -> Result<InferenceStream> {
        let engine = self.engine.clone();
        let request_id = request.request_id.clone();
        let model_id = descriptor.id.clone();

        Ok(stream::once(async move {
            let response = tokio::task::spawn_blocking(move || {
                let mold_request = map_image_generation_request(&descriptor, &request);
                let mut engine = engine.lock().map_err(|_| Error::Runtime {
                    message: "mold runtime lock poisoned".to_string(),
                })?;
                let response = engine.generate(&mold_request).map_err(map_mold_error)?;
                Ok::<_, Error>(map_image_generation_response(
                    request.request_id,
                    descriptor.id,
                    response,
                ))
            })
            .await;

            Ok(match response {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => map_failure_response(error, request_id, model_id),
                Err(error) => map_failure_response(
                    Error::Runtime {
                        message: format!("failed to join mold request task: {error}"),
                    },
                    request_id,
                    model_id,
                ),
            })
        })
        .boxed())
    }
}

fn build_engine(
    runtime_config: &MoldRuntimeConfig,
    model_name: &str,
    model_config: &MoldModelConfig,
) -> Result<Box<dyn MoldEngine>> {
    match &model_config.loader {
        MoldLoader::Flux2(loader) => Ok(Box::new(Flux2Engine::new(
            model_name.to_string(),
            build_flux2_paths(loader)?,
            runtime_config.qwen3_variant.clone(),
            map_load_strategy(runtime_config.load_strategy),
            runtime_config.gpu_ordinal,
            runtime_config.offload,
            None,
        ))),
    }
}

fn build_flux2_paths(loader: &MoldFlux2Loader) -> Result<MoldModelPaths> {
    build_flux2_paths_from_dir(&resolve_model_storage_dir(&loader.model_id))
}

fn build_flux2_paths_from_dir(model_dir: &Path) -> Result<MoldModelPaths> {
    let transformer_dir = model_dir.join("transformer");
    let transformer_shards = collect_safetensors(&transformer_dir)?;
    let transformer = transformer_shards
        .first()
        .cloned()
        .or_else(|| first_root_flux2_transformer(model_dir))
        .ok_or_else(|| missing_path_error("Flux.2 transformer", model_dir))?;

    let vae = first_existing([
        model_dir.join("ae.safetensors"),
        model_dir
            .join("vae")
            .join("diffusion_pytorch_model.safetensors"),
    ])
    .ok_or_else(|| missing_path_error("Flux.2 VAE", model_dir))?;

    let text_encoder_files = collect_safetensors(&model_dir.join("text_encoder"))?;
    if text_encoder_files.is_empty() {
        return Err(missing_path_error("Flux.2 text encoder", model_dir));
    }

    let text_tokenizer = first_existing([
        model_dir.join("tokenizer").join("tokenizer.json"),
        model_dir.join("tokenizer.json"),
    ])
    .ok_or_else(|| missing_path_error("Flux.2 tokenizer", model_dir))?;

    Ok(MoldModelPaths {
        transformer,
        transformer_shards,
        vae,
        spatial_upscaler: None,
        temporal_upscaler: None,
        distilled_lora: None,
        t5_encoder: None,
        clip_encoder: None,
        t5_tokenizer: None,
        clip_tokenizer: None,
        clip_encoder_2: None,
        clip_tokenizer_2: None,
        text_encoder_files,
        text_tokenizer: Some(text_tokenizer),
        decoder: None,
    })
}

fn collect_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") && path.is_file()
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn first_root_flux2_transformer(model_dir: &Path) -> Option<PathBuf> {
    let mut candidates = fs::read_dir(model_dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name != "ae.safetensors" && name.starts_with("flux"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

fn first_existing<const N: usize>(paths: [PathBuf; N]) -> Option<PathBuf> {
    paths.into_iter().find(|path| path.exists())
}

fn missing_path_error(component: &str, model_dir: &Path) -> Error {
    Error::Runtime {
        message: format!(
            "{component} was not found under local model directory `{}`",
            model_dir.display()
        ),
    }
}

fn map_image_generation_request(
    descriptor: &ModelDefinition,
    request: &ImageGenerationRequest,
) -> MoldGenerateRequest {
    MoldGenerateRequest {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        model: descriptor.id.to_string(),
        width: request.size.width,
        height: request.size.height,
        steps: request.steps.unwrap_or(DEFAULT_FLUX2_STEPS),
        guidance: request.guidance_scale.map(f64::from).unwrap_or(0.0),
        seed: request.seed,
        batch_size: request.sample_count,
        output_format: Some(mold_ai_core::OutputFormat::Png),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        edit_images: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
    }
}

fn map_image_generation_response(
    request_id: Option<RequestId>,
    model_id: ModelId,
    response: mold_ai_core::GenerateResponse,
) -> InferenceResponse {
    let images = response
        .images
        .into_iter()
        .map(map_generated_image)
        .collect();

    InferenceResponse::Images(nexo_core::ImageGenerationResponse {
        request_id,
        model_id: Some(model_id),
        images,
    })
}

fn map_generated_image(image: mold_ai_core::ImageData) -> GeneratedImage {
    GeneratedImage {
        index: image.index as usize,
        source: MediaSource::Bytes(image.data),
        media_type: Some(image.format.content_type().to_string()),
        width: Some(image.width),
        height: Some(image.height),
    }
}

fn map_failure_response(
    error: Error,
    request_id: Option<RequestId>,
    _model_id: ModelId,
) -> InferenceResponse {
    let (code, retryability) = match error {
        Error::UnsupportedRequest { .. } | Error::UnsupportedFeature { .. } => {
            (InferenceErrorCode::UnsupportedFeature, Retryability::Fatal)
        }
        Error::UnknownModel { .. } | Error::ModelNotLoaded { .. } => {
            (InferenceErrorCode::ModelUnavailable, Retryability::Fatal)
        }
        Error::Json(_) => (InferenceErrorCode::InvalidRequest, Retryability::Fatal),
        Error::Core(_) => (InferenceErrorCode::InvalidRequest, Retryability::Fatal),
        Error::EmptyModelCatalog
        | Error::DuplicateModelId { .. }
        | Error::UnresolvedModelSelection { .. }
        | Error::UnsupportedMessagePart { .. }
        | Error::InvalidToolPayload { .. }
        | Error::Config { .. }
        | Error::Runtime { .. }
        | Error::Io(_) => (InferenceErrorCode::Internal, Retryability::Retryable),
    };

    InferenceResponse::Failure(InferenceFailure {
        request_id,
        run_id: None,
        round_id: None,
        code,
        message: error.to_string(),
        retryability,
    })
}

fn map_mold_error(error: mold_ai_inference::InferenceError) -> Error {
    Error::Runtime {
        message: error.to_string(),
    }
}

fn map_load_strategy(strategy: ConfigLoadStrategy) -> MoldLoadStrategy {
    match strategy {
        ConfigLoadStrategy::Eager => MoldLoadStrategy::Eager,
        ConfigLoadStrategy::Sequential => MoldLoadStrategy::Sequential,
    }
}

fn request_kind(request: &InferenceRequest) -> &'static str {
    match request {
        InferenceRequest::Generate(_) => "generate",
        InferenceRequest::Embed(_) => "embed",
        InferenceRequest::GenerateImage(_) => "generate_image",
        InferenceRequest::GenerateSpeech(_) => "generate_speech",
        InferenceRequest::Tokenize(_) => "tokenize",
        InferenceRequest::Detokenize(_) => "detokenize",
    }
}

fn mold_runtime_config(runtime_config: &RuntimeConfig) -> MoldRuntimeConfig {
    runtime_config
        .runtime(InferenceRuntime::Mold)
        .and_then(|implementation| implementation.as_mold())
        .cloned()
        .unwrap_or_default()
}

fn mold_model_config(model: &RegisteredModelConfig) -> Result<&MoldModelConfig> {
    model
        .runtime(InferenceRuntime::Mold)
        .and_then(|implementation| implementation.as_mold())
        .ok_or_else(|| Error::UnsupportedFeature {
            feature: format!("model `{}` is not configured for mold", model.descriptor.id),
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used)]

    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{}-{unique}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"test").unwrap();
    }

    #[test]
    fn build_flux2_paths_uses_diffusers_layout() {
        let dir = temp_dir("nexo-mold-flux2-paths");
        touch(&dir.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors"));
        touch(&dir.join("transformer/diffusion_pytorch_model-00002-of-00002.safetensors"));
        touch(&dir.join("vae/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("text_encoder/model-00001-of-00002.safetensors"));
        touch(&dir.join("text_encoder/model-00002-of-00002.safetensors"));
        touch(&dir.join("tokenizer/tokenizer.json"));

        let paths = build_flux2_paths_from_dir(&dir).unwrap();

        assert_eq!(paths.transformer_shards.len(), 2);
        assert_eq!(paths.transformer, paths.transformer_shards[0]);
        assert_eq!(
            paths.vae,
            dir.join("vae/diffusion_pytorch_model.safetensors")
        );
        assert_eq!(
            paths.text_tokenizer.as_deref(),
            Some(dir.join("tokenizer/tokenizer.json").as_path())
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn build_flux2_paths_falls_back_to_root_aliases() {
        let dir = temp_dir("nexo-mold-flux2-root");
        touch(&dir.join("flux-2-klein-9b.safetensors"));
        touch(&dir.join("ae.safetensors"));
        touch(&dir.join("text_encoder/model.safetensors"));
        touch(&dir.join("tokenizer/tokenizer.json"));

        let paths = build_flux2_paths_from_dir(&dir).unwrap();

        assert!(paths.transformer_shards.is_empty());
        assert_eq!(paths.transformer, dir.join("flux-2-klein-9b.safetensors"));
        assert_eq!(paths.vae, dir.join("ae.safetensors"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn map_request_uses_flux2_defaults() {
        let descriptor = ModelDefinition {
            id: "flux.2-klein-9b".into(),
            display_name: "FLUX.2 Klein 9B".to_string(),
            provider: None,
            runtime: nexo_core::InferenceRuntime::Mold,
            capabilities: vec![
                nexo_core::ModelCapability::ImageGeneration,
                nexo_core::ModelCapability::TextGeneration,
            ],
            role_strategy: nexo_core::RoleStrategy::Default,
            context_window_tokens: None,
            max_output_tokens: None,
            metadata: Default::default(),
        };
        let request = ImageGenerationRequest {
            request_id: None,
            session_id: None,
            model: nexo_core::ModelSelection {
                specific_model: None,
                required_capabilities: Vec::new(),
            },
            prompt: "a test image".to_string(),
            negative_prompt: None,
            size: nexo_core::ImageGenerationSize {
                width: 1024,
                height: 768,
            },
            sample_count: 2,
            steps: None,
            guidance_scale: None,
            seed: Some(7),
            metadata: Default::default(),
        };

        let mapped = map_image_generation_request(&descriptor, &request);

        assert_eq!(mapped.model, "flux.2-klein-9b");
        assert_eq!(mapped.steps, DEFAULT_FLUX2_STEPS);
        assert_eq!(mapped.guidance, 0.0);
        assert_eq!(mapped.batch_size, 2);
        assert_eq!(mapped.output_format, Some(mold_ai_core::OutputFormat::Png));
    }

    #[test]
    fn missing_runtime_config_uses_mold_defaults() {
        let runtime = mold_runtime_config(&RuntimeConfig {
            scheduler: crate::engine::config::SchedulerPolicy::default(),
            runtimes: Vec::new(),
        });

        assert_eq!(runtime.gpu_ordinal, 0);
        assert_eq!(runtime.load_strategy, ConfigLoadStrategy::Sequential);
        assert!(!runtime.offload);
        assert_eq!(runtime.qwen3_variant, None);
    }
}
