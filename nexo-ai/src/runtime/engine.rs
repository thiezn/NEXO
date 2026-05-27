use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;
use futures_util::{StreamExt, stream};
use mistralrs_core::{
    AddModelConfig, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting,
    EngineConfig as MistralEngineConfig, GGUFLoaderBuilder, GGUFSpecificConfig, LoaderBuilder,
    MistralRs, MistralRsBuilder, ModelSelected, ModelStatus, Request,
    SchedulerConfig as MistralSchedulerConfig, TokenSource, get_auto_device_map_params,
    get_model_dtype,
};
use nexo_core::inference::request::{EmbedRequest, GenerateRequest};
use nexo_core::{
    DetokenizationRequest, InferenceRequest, InferenceResponse, InferenceStream, ModelDescriptor,
    ModelId, ModelRuntimeState, TokenUsage, TokenizationRequest,
};
use tokio::sync::mpsc;

use crate::config::{
    AutoModelLoader, DeviceSpec, GgufModelLoader, ModelDataType, ModelLoader,
    RegisteredModelConfig, RuntimeConfig, SchedulerPolicy,
};
use crate::mapping::request::{
    map_detokenization_request, map_embedding_request, map_generate_request,
    map_tokenization_request,
};
use crate::mapping::response::{
    ResponseContext, generation_started, map_embedding_response, map_generation_response,
    map_runtime_error,
};
use crate::{Error, NexoAiConfig, Result};

/// Shared `mistralrs-core` runtime state used by `NexoAi`.
#[derive(Clone)]
pub(crate) struct MistralRuntime {
    engine: Arc<MistralRs>,
    next_request_ordinal: Arc<AtomicUsize>,
}

impl std::fmt::Debug for MistralRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRuntime").finish_non_exhaustive()
    }
}

impl MistralRuntime {
    /// Builds the backing `mistralrs-core` runtime from declarative configuration.
    pub(crate) async fn from_config(config: &NexoAiConfig) -> Result<Self> {
        let device = resolve_device(config.runtime.device)?;
        let mut models = config.models.iter();
        let first = models.next().ok_or(Error::EmptyModelCatalog)?;

        let first_pipeline = build_pipeline(first, &config.runtime, &device)?;
        let scheduler = map_scheduler(config.runtime.scheduler);
        let engine = MistralRsBuilder::new(
            first_pipeline,
            scheduler.clone(),
            config.runtime.throughput_logging,
            None,
        )
        .with_model_id(first.descriptor.id.to_string())
        .with_no_kv_cache(config.runtime.no_kv_cache)
        .with_no_prefix_cache(config.runtime.no_prefix_cache)
        .with_prefix_cache_n(config.runtime.prefix_cache_entries)
        .with_disable_eos_stop(config.runtime.disable_eos_stop)
        .build()
        .await;

        for model in models {
            let pipeline = build_pipeline(model, &config.runtime, &device)?;
            engine
                .add_model(
                    model.descriptor.id.to_string(),
                    pipeline,
                    scheduler.clone(),
                    AddModelConfig::new(map_engine_config(&config.runtime)),
                )
                .await
                .map_err(|message| Error::MistralRuntime { message })?;
        }

        Ok(Self {
            engine,
            next_request_ordinal: Arc::new(AtomicUsize::new(1)),
        })
    }

    /// Returns the runtime state for a configured model, if it is known by `mistralrs-core`.
    pub(crate) fn model_state(&self, model_id: &ModelId) -> Option<ModelRuntimeState> {
        self.engine
            .get_model_status(model_id.as_str())
            .ok()
            .flatten()
            .map(|status| match status {
                ModelStatus::Loaded => ModelRuntimeState::Loaded,
                ModelStatus::Unloaded => ModelRuntimeState::Unloaded,
                ModelStatus::Reloading => ModelRuntimeState::Reloading,
            })
    }

    /// Submits a shared `nexo-core` request to the backing runtime.
    pub(crate) fn submit(
        &self,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match request {
            InferenceRequest::Generate(request) => self.submit_generate(descriptor, request),
            InferenceRequest::Embed(request) => self.submit_embed(descriptor, request),
            InferenceRequest::Tokenize(request) => self.submit_tokenize(descriptor, request),
            InferenceRequest::Detokenize(request) => self.submit_detokenize(descriptor, request),
            InferenceRequest::GenerateImage(_) => Err(Error::UnsupportedRequest {
                kind: "generate_image",
            }),
            InferenceRequest::GenerateSpeech(_) => Err(Error::UnsupportedRequest {
                kind: "generate_speech",
            }),
        }
    }

    fn submit_generate(
        &self,
        descriptor: ModelDescriptor,
        request: GenerateRequest,
    ) -> Result<InferenceStream> {
        let context = ResponseContext {
            request_id: request.request_id.clone(),
            run_id: request.run_id.clone(),
            round_id: request.round_id.clone(),
            model_id: descriptor.id.clone(),
        };
        let (response_tx, response_rx) = mpsc::channel(32);
        let mistral_request = map_generate_request(
            &request,
            &descriptor,
            response_tx,
            self.next_request_ordinal(),
        )?;

        self.dispatch_request(&descriptor.id, Request::Normal(Box::new(mistral_request)))?;

        let started = stream::once({
            let context = context.clone();
            async move { Ok(generation_started(&context)) }
        });
        let body = stream::unfold(
            (response_rx, context),
            |(mut response_rx, context)| async move {
                response_rx.recv().await.map(|response| {
                    (
                        Ok(map_generation_response(response, &context)),
                        (response_rx, context),
                    )
                })
            },
        );

        Ok(started.chain(body).boxed())
    }

    fn submit_embed(
        &self,
        descriptor: ModelDescriptor,
        request: EmbedRequest,
    ) -> Result<InferenceStream> {
        let runtime = self.clone();
        let request_id = request.request_id.clone();
        let model_id = descriptor.id.clone();
        let request_ordinal = self.next_request_ordinal();
        Ok(stream::once(async move {
            match runtime
                .execute_embeddings(descriptor, request, request_ordinal)
                .await
            {
                Ok((vectors, usage)) => {
                    Ok(map_embedding_response(request_id, model_id, vectors, usage))
                }
                Err(error) => Ok(map_runtime_error(error, request_id, None, None)),
            }
        })
        .boxed())
    }

    fn submit_tokenize(
        &self,
        descriptor: ModelDescriptor,
        request: TokenizationRequest,
    ) -> Result<InferenceStream> {
        let runtime = self.clone();
        let request_id = request.request_id.clone();
        Ok(stream::once(async move {
            match runtime.execute_tokenization(descriptor, request).await {
                Ok(tokens) => Ok(InferenceResponse::Tokenization(
                    nexo_core::TokenizationResponse { request_id, tokens },
                )),
                Err(error) => Ok(map_runtime_error(error, request_id, None, None)),
            }
        })
        .boxed())
    }

    fn submit_detokenize(
        &self,
        descriptor: ModelDescriptor,
        request: DetokenizationRequest,
    ) -> Result<InferenceStream> {
        let runtime = self.clone();
        let request_id = request.request_id.clone();
        Ok(stream::once(async move {
            match runtime.execute_detokenization(descriptor, request).await {
                Ok(text) => Ok(InferenceResponse::Detokenization(
                    nexo_core::DetokenizationResponse { request_id, text },
                )),
                Err(error) => Ok(map_runtime_error(error, request_id, None, None)),
            }
        })
        .boxed())
    }

    async fn execute_embeddings(
        &self,
        descriptor: ModelDescriptor,
        request: EmbedRequest,
        first_request_ordinal: usize,
    ) -> Result<(Vec<nexo_core::EmbeddingVector>, Option<TokenUsage>)> {
        let mut vectors = Vec::with_capacity(request.inputs.len());
        let mut aggregated_usage = TokenUsage::default();

        for (index, input) in request.inputs.into_iter().enumerate() {
            let (response_tx, mut response_rx) = mpsc::channel(1);
            let request_ordinal = if index == 0 {
                first_request_ordinal
            } else {
                self.next_request_ordinal()
            };
            let mistral_request =
                map_embedding_request(input, &descriptor, response_tx, request_ordinal);
            self.dispatch_request(&descriptor.id, Request::Normal(Box::new(mistral_request)))?;

            let Some(response) = response_rx.recv().await else {
                return Err(Error::MistralRuntime {
                    message: "embedding response channel closed before producing output"
                        .to_string(),
                });
            };

            match response {
                mistralrs_core::Response::Embeddings {
                    embeddings,
                    prompt_tokens,
                    total_tokens,
                } => {
                    vectors.push(nexo_core::EmbeddingVector {
                        index,
                        values: embeddings,
                    });
                    aggregated_usage.input_tokens += prompt_tokens;
                    aggregated_usage.total_tokens += total_tokens;
                }
                other => {
                    return Err(Error::MistralRuntime {
                        message: format!(
                            "unexpected embedding response variant: {}",
                            other
                                .as_result()
                                .err()
                                .map(|error| error.to_string())
                                .unwrap_or_else(|| "non-embedding output".to_string())
                        ),
                    });
                }
            }
        }

        let usage = if vectors.is_empty() {
            None
        } else {
            Some(aggregated_usage)
        };

        Ok((vectors, usage))
    }

    async fn execute_tokenization(
        &self,
        descriptor: ModelDescriptor,
        request: TokenizationRequest,
    ) -> Result<Vec<u32>> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        let mistral_request = map_tokenization_request(&request, &descriptor, response_tx)?;
        self.dispatch_request(&descriptor.id, Request::Tokenize(mistral_request))?;

        match response_rx.recv().await {
            Some(Ok(tokens)) => Ok(tokens),
            Some(Err(error)) => Err(Error::MistralRuntime {
                message: error.to_string(),
            }),
            None => Err(Error::MistralRuntime {
                message: "tokenization response channel closed before producing output".to_string(),
            }),
        }
    }

    async fn execute_detokenization(
        &self,
        descriptor: ModelDescriptor,
        request: DetokenizationRequest,
    ) -> Result<String> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        let mistral_request = map_detokenization_request(&request, response_tx);
        self.dispatch_request(&descriptor.id, Request::Detokenize(mistral_request))?;

        match response_rx.recv().await {
            Some(Ok(text)) => Ok(text),
            Some(Err(error)) => Err(Error::MistralRuntime {
                message: error.to_string(),
            }),
            None => Err(Error::MistralRuntime {
                message: "detokenization response channel closed before producing output"
                    .to_string(),
            }),
        }
    }

    fn dispatch_request(&self, model_id: &ModelId, request: Request) -> Result {
        let sender = self
            .engine
            .get_sender(Some(model_id.as_str()))
            .map_err(|error| Error::MistralRuntime {
                message: error.to_string(),
            })?;
        sender
            .blocking_send(request)
            .map_err(|_| Error::MistralRuntime {
                message: format!("failed to dispatch request to model `{model_id}`"),
            })
    }

    fn next_request_ordinal(&self) -> usize {
        self.next_request_ordinal.fetch_add(1, Ordering::Relaxed)
    }
}

fn build_pipeline(
    model: &RegisteredModelConfig,
    runtime_config: &RuntimeConfig,
    device: &Device,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    match &model.loader {
        ModelLoader::Auto(loader) => {
            build_auto_pipeline(loader, &model.revision, runtime_config, device)
        }
        ModelLoader::Gguf(loader) => {
            build_gguf_pipeline(loader, &model.revision, runtime_config, device)
        }
    }
}

fn build_auto_pipeline(
    loader: &AutoModelLoader,
    revision: &Option<String>,
    runtime_config: &RuntimeConfig,
    device: &Device,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    let selected = ModelSelected::Run {
        model_id: loader.model_id.clone(),
        tokenizer_json: path_to_string(loader.tokenizer_json.as_ref()),
        dtype: map_dtype(loader.dtype),
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_edge: None,
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        max_num_images: None,
        max_image_length: None,
        hf_cache_path: loader.hf_cache_path.clone(),
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let built_loader = LoaderBuilder::new(selected.clone())
        .with_no_kv_cache(runtime_config.no_kv_cache)
        .with_chat_template(path_to_string(loader.chat_template.as_ref()))
        .with_jinja_explicit(path_to_string(loader.jinja_explicit.as_ref()))
        .build()
        .map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })?;

    let dtype = get_model_dtype(&selected).map_err(|error| Error::MistralRuntime {
        message: error.to_string(),
    })?;
    let device_map =
        get_auto_device_map_params(&selected).map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })?;

    built_loader
        .load_model_from_hf(
            revision.clone(),
            TokenSource::CacheToken,
            &dtype,
            device,
            true,
            DeviceMapSetting::Auto(device_map),
            None,
            None,
        )
        .map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })
}

fn build_gguf_pipeline(
    loader: &GgufModelLoader,
    revision: &Option<String>,
    runtime_config: &RuntimeConfig,
    device: &Device,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    let built_loader = GGUFLoaderBuilder::new(
        path_to_string(loader.chat_template.as_ref()),
        loader.tokenizer_model_id.clone(),
        loader.quantized_model_id.clone(),
        loader.quantized_filenames.clone(),
        GGUFSpecificConfig::default(),
        runtime_config.no_kv_cache,
        path_to_string(loader.jinja_explicit.as_ref()),
    )
    .build();

    built_loader
        .load_model_from_hf(
            revision.clone(),
            TokenSource::CacheToken,
            &map_dtype(loader.dtype),
            device,
            true,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            None,
        )
        .map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })
}

fn resolve_device(device: DeviceSpec) -> Result<Device> {
    match device {
        DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Metal => metal_device(),
        DeviceSpec::BestAvailable => best_available_device(),
    }
}

fn best_available_device() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }

    Ok(Device::Cpu)
}

#[cfg(feature = "metal")]
fn metal_device() -> Result<Device> {
    Device::new_metal(0).map_err(|error| Error::MistralRuntime {
        message: error.to_string(),
    })
}

#[cfg(not(feature = "metal"))]
fn metal_device() -> Result<Device> {
    Err(Error::UnsupportedFeature {
        feature: "metal backend requested but the crate was built without the `metal` feature"
            .to_string(),
    })
}

fn map_scheduler(policy: SchedulerPolicy) -> MistralSchedulerConfig {
    match policy {
        SchedulerPolicy::Fixed {
            max_running_sequences,
        } => MistralSchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(max_running_sequences),
        },
    }
}

fn map_engine_config(runtime_config: &RuntimeConfig) -> MistralEngineConfig {
    MistralEngineConfig {
        no_kv_cache: runtime_config.no_kv_cache,
        no_prefix_cache: runtime_config.no_prefix_cache,
        prefix_cache_n: runtime_config.prefix_cache_entries,
        disable_eos_stop: runtime_config.disable_eos_stop,
        throughput_logging_enabled: runtime_config.throughput_logging,
        ..MistralEngineConfig::default()
    }
}

fn map_dtype(dtype: ModelDataType) -> mistralrs_core::ModelDType {
    match dtype {
        ModelDataType::Auto => mistralrs_core::ModelDType::Auto,
        ModelDataType::Bf16 => mistralrs_core::ModelDType::BF16,
        ModelDataType::F16 => mistralrs_core::ModelDType::F16,
        ModelDataType::F32 => mistralrs_core::ModelDType::F32,
    }
}

fn path_to_string(path: Option<&PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().into_owned())
}
