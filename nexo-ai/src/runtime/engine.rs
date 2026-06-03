use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;
use futures_util::{StreamExt, stream};
use mistralrs_core::{
    AddModelConfig, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting,
    EngineConfig as MistralEngineConfig, GGUFLoaderBuilder, GGUFSpecificConfig, LoaderBuilder,
    MemoryGpuConfig, MistralRs, MistralRsBuilder, ModelPaths, ModelSelected,
    PagedAttentionConfig, PagedCacheType, Request, Response as MistralResponse,
    SchedulerConfig as MistralSchedulerConfig, TokenSource, UQFF_MULTI_FILE_DELIMITER,
    paged_attn_supported,
    get_auto_device_map_params, get_model_dtype,
};
use nexo_core::inference::request::{EmbedRequest, GenerateRequest};
use nexo_core::{
    DetokenizationRequest, InferenceRequest, InferenceResponse, InferenceStream, ModelDescriptor,
    ModelId, TokenUsage, TokenizationRequest,
};
use nexo_model_mgmt::resolve_model_storage_dir;
use tokio::sync::mpsc;

use crate::config::{
    AutoModelLoader, DeviceSpec, GgufModelLoader, ModelDataType, ModelLoader,
    PagedAttentionCacheType, PagedAttentionMode, RegisteredModelConfig, RuntimeConfig,
    SchedulerPolicy,
};
use crate::mapping::request::{
    map_detokenization_request, map_embedding_request, map_generate_request,
    map_tokenization_request,
};
use crate::mapping::response::{
    ResponseContext, generation_started, map_embedding_response, map_generation_response,
    map_runtime_error,
};
use crate::{Error, Result};

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
    pub(crate) async fn from_model_config(
        base_runtime_config: &RuntimeConfig,
        model: &RegisteredModelConfig,
    ) -> Result<Self> {
        let device = resolve_device(base_runtime_config.device)?;
        Self::from_models(base_runtime_config, &device, std::slice::from_ref(model)).await
    }

    async fn from_models(
        runtime_config: &RuntimeConfig,
        device: &Device,
        models: &[RegisteredModelConfig],
    ) -> Result<Self> {
        let mut models = models.iter();
        let first = models.next().ok_or(Error::EmptyModelCatalog)?;
        let paged_attn_config = resolve_paged_attention_config(runtime_config, device)?;

        let first_pipeline = build_pipeline(first, runtime_config, device, paged_attn_config)?;
        let scheduler =
            map_scheduler(runtime_config.scheduler, &first_pipeline, paged_attn_config).await;
        let engine = MistralRsBuilder::new(
            first_pipeline,
            scheduler.clone(),
            runtime_config.throughput_logging,
            None,
        )
        .with_model_id(first.descriptor.id.to_string())
        .with_no_kv_cache(runtime_config.no_kv_cache)
        .with_no_prefix_cache(runtime_config.no_prefix_cache)
        .with_prefix_cache_n(runtime_config.prefix_cache_entries)
        .with_disable_eos_stop(runtime_config.disable_eos_stop)
        .build()
        .await;

        for model in models {
            let pipeline = build_pipeline(model, runtime_config, device, paged_attn_config)?;
            engine
                .add_model(
                    model.descriptor.id.to_string(),
                    pipeline,
                    scheduler.clone(),
                    AddModelConfig::new(map_engine_config(&runtime_config)),
                )
                .await
                .map_err(|message| Error::MistralRuntime { message })?;
        }

        Ok(Self {
            engine,
            next_request_ordinal: Arc::new(AtomicUsize::new(1)),
        })
    }

    /// Submits a shared `nexo-core` request to the backing runtime.
    pub(crate) async fn submit(
        &self,
        descriptor: ModelDescriptor,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        match request {
            InferenceRequest::Generate(request) => self.submit_generate(descriptor, request).await,
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

    async fn submit_generate(
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
        let request_ordinal = self.next_request_ordinal();
        let text_parts_per_message = request
            .conversation
            .messages
            .iter()
            .map(text_part_count)
            .collect::<Vec<_>>();
        let text_parts_total = text_parts_per_message.iter().sum::<usize>();
        let messages_with_multiple_text_parts = text_parts_per_message
            .iter()
            .filter(|count| **count > 1)
            .count();
        tracing::debug!(
            request_ordinal,
            model_id = %descriptor.id,
            request_id = ?request.request_id,
            run_id = ?request.run_id,
            round_id = ?request.round_id,
            session_id = ?request.session_id,
            messages = request.conversation.messages.len(),
            text_parts_total,
            messages_with_multiple_text_parts,
            text_parts_per_message = ?text_parts_per_message,
            tools = request.tools.len(),
            streaming = ?request.streaming,
            sampling = ?request.sampling,
            "Submitting generate request to mistralrs"
        );

        let mistral_request =
            map_generate_request(&request, &descriptor, response_tx, request_ordinal)?;

        if let Err(error) = self
            .dispatch_request(&descriptor.id, Request::Normal(Box::new(mistral_request)))
            .await
        {
            tracing::debug!(
                request_ordinal,
                model_id = %descriptor.id,
                error = %error,
                "Failed to dispatch generate request to mistralrs"
            );
            return Err(error);
        }
        tracing::debug!(
            request_ordinal,
            model_id = %descriptor.id,
            "Dispatched generate request to mistralrs"
        );

        let started = stream::once({
            let context = context.clone();
            async move { Ok(generation_started(&context)) }
        });
        let body = stream::unfold(
            (response_rx, context, request_ordinal),
            |(mut response_rx, context, request_ordinal)| async move {
                match response_rx.recv().await {
                    Some(response) => {
                        tracing::debug!(
                            request_ordinal,
                            model_id = %context.model_id,
                            response_kind = mistral_response_kind(&response),
                            "Received mistralrs generation response"
                        );
                        if let Some(error) = mistral_response_error(&response) {
                            tracing::debug!(
                                request_ordinal,
                                model_id = %context.model_id,
                                error = %error,
                                "Received mistralrs generation error response"
                            );
                        }
                        Some((
                            Ok(map_generation_response(response, &context)),
                            (response_rx, context, request_ordinal),
                        ))
                    }
                    None => {
                        tracing::debug!(
                            request_ordinal,
                            model_id = %context.model_id,
                            "Mistralrs generation response channel closed"
                        );
                        None
                    }
                }
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
            self.dispatch_request(&descriptor.id, Request::Normal(Box::new(mistral_request)))
                .await?;

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
        self.dispatch_request(&descriptor.id, Request::Tokenize(mistral_request))
            .await?;

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
        self.dispatch_request(&descriptor.id, Request::Detokenize(mistral_request))
            .await?;

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

    async fn dispatch_request(&self, model_id: &ModelId, request: Request) -> Result {
        let sender = self
            .engine
            .get_sender(Some(model_id.as_str()))
            .map_err(|error| Error::MistralRuntime {
                message: error.to_string(),
            })?;
        sender
            .send(request)
            .await
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
    paged_attn_config: Option<PagedAttentionConfig>,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    match &model.loader {
        ModelLoader::Auto(loader) => {
            build_auto_pipeline(loader, runtime_config, device, paged_attn_config)
        }
        ModelLoader::Gguf(loader) => {
            build_gguf_pipeline(loader, runtime_config, device, paged_attn_config)
        }
    }
}

fn build_auto_pipeline(
    loader: &AutoModelLoader,
    runtime_config: &RuntimeConfig,
    device: &Device,
    paged_attn_config: Option<PagedAttentionConfig>,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    let model_dir = resolve_model_storage_dir(&loader.model_id);
    let from_uqff = resolve_uqff_selection(&model_dir, loader.from_uqff.as_deref())?;
    let selected_model_id = if from_uqff.is_some() {
        model_dir.to_string_lossy().into_owned()
    } else {
        loader.model_id.clone()
    };
    let selected = ModelSelected::Run {
        model_id: selected_model_id,
        tokenizer_json: path_to_string(loader.tokenizer_json.as_ref()),
        dtype: map_dtype(loader.dtype),
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff,
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

    if selected_uses_uqff(&selected) {
        return built_loader
            .load_model_from_hf(
                None,
                TokenSource::None,
                &dtype,
                device,
                true,
                DeviceMapSetting::Auto(device_map),
                None,
                paged_attn_config,
            )
            .map_err(|error| Error::MistralRuntime {
                message: error.to_string(),
            });
    }

    let local_paths = build_auto_model_paths(loader)?;

    built_loader
        .load_model_from_path(
            &local_paths,
            &dtype,
            device,
            true,
            DeviceMapSetting::Auto(device_map),
            None,
            paged_attn_config,
        )
        .map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })
}

fn build_gguf_pipeline(
    loader: &GgufModelLoader,
    runtime_config: &RuntimeConfig,
    device: &Device,
    paged_attn_config: Option<PagedAttentionConfig>,
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

    let local_paths = build_gguf_model_paths(loader)?;

    built_loader
        .load_model_from_path(
            &local_paths,
            &map_dtype(loader.dtype),
            device,
            true,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            paged_attn_config,
        )
        .map_err(|error| Error::MistralRuntime {
            message: error.to_string(),
        })
}

fn selected_uses_uqff(selected: &ModelSelected) -> bool {
    matches!(
        selected,
        ModelSelected::Run {
            from_uqff: Some(_),
            ..
        }
    )
}

fn resolve_uqff_selection(
    model_dir: &Path,
    explicit: Option<&[PathBuf]>,
) -> Result<Option<String>> {
    let files = if let Some(files) = explicit {
        resolve_explicit_uqff_files(model_dir, files)?
    } else if model_dir.exists() {
        discover_local_uqff_files(model_dir)?
    } else {
        Vec::new()
    };

    if files.is_empty() {
        Ok(None)
    } else {
        Ok(Some(files.join(UQFF_MULTI_FILE_DELIMITER)))
    }
}

fn resolve_explicit_uqff_files(model_dir: &Path, selectors: &[PathBuf]) -> Result<Vec<String>> {
    let mut files = Vec::new();
    let mut discovered = None;

    for selector in selectors {
        if is_uqff_file(selector) {
            files.push(selector.to_string_lossy().into_owned());
            continue;
        }

        let prefix = selector.to_string_lossy();
        if discovered.is_none() {
            discovered = Some(discover_local_uqff_files(model_dir)?);
        }
        let local_files = discovered.as_ref().ok_or_else(|| Error::MistralRuntime {
            message: "failed to discover local UQFF files".to_string(),
        })?;
        let mut matched = local_files
            .iter()
            .filter(|filename| filename.starts_with(prefix.as_ref()))
            .cloned()
            .collect::<Vec<_>>();

        if matched.is_empty() {
            return Err(missing_local_file(model_dir, &format!("{prefix}*.uqff")));
        }

        files.append(&mut matched);
    }

    files.sort();
    files.dedup();
    Ok(files)
}

fn discover_local_uqff_files(model_dir: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(model_dir).map_err(|error| Error::MistralRuntime {
        message: format!(
            "failed to read local model directory `{}`: {error}",
            model_dir.display()
        ),
    })?;
    let mut files = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| Error::MistralRuntime {
            message: format!(
                "failed to read local model directory entry under `{}`: {error}",
                model_dir.display()
            ),
        })?;
        let path = entry.path();
        if is_uqff_file(&path)
            && let Some(filename) = path.file_name().and_then(|filename| filename.to_str())
        {
            files.push(filename.to_string());
        }
    }

    files.sort();
    Ok(files)
}

fn build_auto_model_paths(loader: &AutoModelLoader) -> Result<Box<dyn ModelPaths>> {
    let model_dir = resolve_model_storage_dir(&loader.model_id);
    let tokenizer_filename = loader
        .tokenizer_json
        .clone()
        .or_else(|| first_existing(&model_dir, &["tokenizer.json", "tekken.json"]))
        .ok_or_else(|| missing_local_file(&model_dir, "tokenizer.json or tekken.json"))?;
    let config_filename = first_existing(&model_dir, &["config.json", "params.json"])
        .ok_or_else(|| missing_local_file(&model_dir, "config.json or params.json"))?;
    let filenames = collect_weight_files(&model_dir, &[])?;

    Ok(Box::new(mistralrs_core::LocalModelPaths {
        tokenizer_filename,
        config_filename,
        template_filename: loader.chat_template.clone().or_else(|| {
            first_existing(
                &model_dir,
                &["chat_template.jinja", "tokenizer_config.json"],
            )
        }),
        filenames,
        adapter_paths: mistralrs_core::AdapterPaths::None,
        gen_conf: first_existing(&model_dir, &["generation_config.json"]),
        preprocessor_config: first_existing(&model_dir, &["preprocessor_config.json"]),
        processor_config: first_existing(&model_dir, &["processor_config.json"]),
        chat_template_json_filename: loader
            .jinja_explicit
            .clone()
            .or_else(|| first_existing(&model_dir, &["chat_template.json"])),
    }))
}

fn build_gguf_model_paths(loader: &GgufModelLoader) -> Result<Box<dyn ModelPaths>> {
    let model_dir = resolve_model_storage_dir(&loader.quantized_model_id);
    let filenames = loader
        .quantized_filenames
        .iter()
        .map(|filename| resolve_local_file(&model_dir, filename))
        .collect::<Result<Vec<_>>>()?;
    let tokenizer_dir = loader
        .tokenizer_model_id
        .as_ref()
        .map(|model_id| resolve_model_storage_dir(model_id))
        .unwrap_or_else(|| model_dir.clone());

    Ok(Box::new(mistralrs_core::LocalModelPaths {
        tokenizer_filename: first_existing(&tokenizer_dir, &["tokenizer.json"]).unwrap_or_default(),
        config_filename: first_existing(&tokenizer_dir, &["config.json"]).unwrap_or_default(),
        template_filename: loader.chat_template.clone().or_else(|| {
            first_existing(
                &tokenizer_dir,
                &["chat_template.jinja", "tokenizer_config.json"],
            )
        }),
        filenames,
        adapter_paths: mistralrs_core::AdapterPaths::None,
        gen_conf: first_existing(&tokenizer_dir, &["generation_config.json"]),
        preprocessor_config: first_existing(&tokenizer_dir, &["preprocessor_config.json"]),
        processor_config: first_existing(&tokenizer_dir, &["processor_config.json"]),
        chat_template_json_filename: loader
            .jinja_explicit
            .clone()
            .or_else(|| first_existing(&tokenizer_dir, &["chat_template.json"])),
    }))
}

fn collect_weight_files(model_dir: &Path, explicit_filenames: &[String]) -> Result<Vec<PathBuf>> {
    if !explicit_filenames.is_empty() {
        return explicit_filenames
            .iter()
            .map(|filename| resolve_local_file(model_dir, filename))
            .collect();
    }

    let mut files = Vec::new();
    collect_weight_files_recursive(model_dir, &mut files)?;
    files.sort();

    if files.is_empty() {
        return Err(Error::MistralRuntime {
            message: format!(
                "no local model weight files found under `{}`; run `nexo-ai models pull <model>` first",
                model_dir.display()
            ),
        });
    }

    Ok(files)
}

fn collect_weight_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result {
    let entries = std::fs::read_dir(dir).map_err(|error| Error::MistralRuntime {
        message: format!(
            "failed to read local model directory `{}`: {error}",
            dir.display()
        ),
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| Error::MistralRuntime {
            message: format!(
                "failed to read local model directory entry under `{}`: {error}",
                dir.display()
            ),
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_weight_files_recursive(&path, files)?;
        } else if is_weight_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

fn resolve_local_file(model_dir: &Path, filename: &str) -> Result<PathBuf> {
    let path = Path::new(filename);
    let resolved = if path.is_absolute() || path.exists() {
        path.to_path_buf()
    } else {
        model_dir.join(path)
    };

    if resolved.exists() {
        Ok(resolved)
    } else {
        Err(missing_local_file(model_dir, filename))
    }
}

fn first_existing(model_dir: &Path, filenames: &[&str]) -> Option<PathBuf> {
    filenames
        .iter()
        .map(|filename| model_dir.join(filename))
        .find(|path| path.exists())
}

fn is_weight_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "safetensors" | "bin" | "pth" | "pt"))
}

fn is_uqff_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "uqff")
}

fn missing_local_file(model_dir: &Path, filename: &str) -> Error {
    Error::MistralRuntime {
        message: format!(
            "missing local model file `{}` under `{}`; run `nexo-ai models pull <model>` first",
            filename,
            model_dir.display()
        ),
    }
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::catalog::model_config_from_manifest;
    use nexo_model_mgmt::registry::find_manifest;

    use super::*;

    #[test]
    fn gemma_4_uqff_keeps_prefix_cache_default() {
        let _model =
            model_config_from_manifest(find_manifest("gemma-4-e4b-it-uqff-afq8").unwrap()).unwrap();
        let runtime = RuntimeConfig::default();

        assert!(!runtime.no_prefix_cache);
    }

    #[test]
    fn gemma_4_gguf_keeps_prefix_cache_default() {
        let _model =
            model_config_from_manifest(find_manifest("gemma-4-e2b-it-q5").unwrap()).unwrap();
        let runtime = RuntimeConfig::default();

        assert!(!runtime.no_prefix_cache);
    }

    #[test]
    fn explicit_uqff_prefix_resolves_only_matching_files() {
        let model_dir = unique_test_dir("uqff-prefix");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("afq6-0.uqff"), []).unwrap();
        fs::write(model_dir.join("afq8-0.uqff"), []).unwrap();

        let selected = resolve_uqff_selection(&model_dir, Some(&[PathBuf::from("afq8-")]))
            .unwrap()
            .unwrap();

        assert_eq!(selected, "afq8-0.uqff");
        fs::remove_dir_all(&model_dir).unwrap();
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("nexo-ai-{name}-{}-{timestamp}", std::process::id()))
    }
}

async fn map_scheduler(
    policy: SchedulerPolicy,
    first_pipeline: &Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>,
    paged_attn_config: Option<PagedAttentionConfig>,
) -> MistralSchedulerConfig {
    match policy {
        SchedulerPolicy::Fixed {
            max_running_sequences,
        } => {
            if paged_attn_config.is_some() {
                let cache_config = first_pipeline.lock().await.get_metadata().cache_config.clone();
                if let Some(config) = cache_config {
                    return MistralSchedulerConfig::PagedAttentionMeta {
                        max_num_seqs: max_running_sequences.get(),
                        config,
                    };
                }

                tracing::warn!(
                    "PagedAttention requested but pipeline metadata did not provide a cache config; falling back to default scheduler"
                );
            }

            MistralSchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(max_running_sequences),
            }
        }
    }
}

fn resolve_paged_attention_config(
    runtime_config: &RuntimeConfig,
    device: &Device,
) -> Result<Option<PagedAttentionConfig>> {
    let requested = match runtime_config.paged_attention.mode {
        PagedAttentionMode::Enabled => true,
        PagedAttentionMode::Disabled => false,
        PagedAttentionMode::Auto => {
            if device.is_cuda() {
                true
            } else {
                false
            }
        }
    };

    if !requested {
        return Ok(None);
    }

    if !paged_attn_supported() {
        tracing::warn!(
            "PagedAttention requested but this build/backend does not support it; continuing without paged attention"
        );
        return Ok(None);
    }

    let memory_config = resolve_paged_memory_config(runtime_config);
    let cache_type = map_paged_cache_type(runtime_config.paged_attention.cache_type);
    let config = PagedAttentionConfig::new(
        runtime_config.paged_attention.block_size,
        memory_config,
        cache_type,
    )
    .map_err(|error| Error::MistralRuntime {
        message: error.to_string(),
    })?;

    Ok(Some(config))
}

fn resolve_paged_memory_config(runtime_config: &RuntimeConfig) -> MemoryGpuConfig {
    let paged = &runtime_config.paged_attention;

    match (
        paged.gpu_memory_mb,
        paged.gpu_memory_utilization,
        paged.context_size,
    ) {
        (_, Some(utilization), Some(_)) => {
            tracing::warn!(
                "Both paged_attention.gpu_memory_utilization and paged_attention.context_size are set; using gpu_memory_utilization"
            );
            MemoryGpuConfig::Utilization(utilization)
        }
        (Some(_), Some(utilization), None) => {
            tracing::warn!(
                "Both paged_attention.gpu_memory_mb and paged_attention.gpu_memory_utilization are set; using gpu_memory_utilization"
            );
            MemoryGpuConfig::Utilization(utilization)
        }
        (Some(_), None, Some(context_size)) => {
            tracing::warn!(
                "Both paged_attention.gpu_memory_mb and paged_attention.context_size are set; using context_size"
            );
            MemoryGpuConfig::ContextSize(context_size)
        }
        (None, Some(utilization), None) => MemoryGpuConfig::Utilization(utilization),
        (Some(memory_mb), None, None) => MemoryGpuConfig::MbAmount(memory_mb),
        (None, None, Some(context_size)) => MemoryGpuConfig::ContextSize(context_size),
        (None, None, None) => MemoryGpuConfig::Utilization(0.9),
    }
}

fn map_paged_cache_type(cache_type: PagedAttentionCacheType) -> PagedCacheType {
    match cache_type {
        PagedAttentionCacheType::Auto => PagedCacheType::Auto,
        PagedAttentionCacheType::F8e4m3 => PagedCacheType::F8E4M3,
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

fn text_part_count(message: &nexo_core::ConversationMessage) -> usize {
    message
        .parts
        .iter()
        .filter(|part| matches!(part, nexo_core::ContentPart::Text(_)))
        .count()
}

fn mistral_response_kind(response: &MistralResponse) -> &'static str {
    match response {
        MistralResponse::Done(_) => "done",
        MistralResponse::Chunk(_) => "chunk",
        MistralResponse::CompletionDone(_) => "completion_done",
        MistralResponse::CompletionChunk(_) => "completion_chunk",
        MistralResponse::Raw { .. } => "raw",
        MistralResponse::Embeddings { .. } => "embeddings",
        MistralResponse::ImageGeneration(_) => "image_generation",
        MistralResponse::Speech { .. } => "speech",
        MistralResponse::AgenticToolCallProgress { .. } => "agentic_tool_call_progress",
        MistralResponse::AgenticToolApprovalRequired { .. } => "agentic_tool_approval_required",
        MistralResponse::File(_) => "file",
        MistralResponse::InternalError(_) => "internal_error",
        MistralResponse::ValidationError(_) => "validation_error",
        MistralResponse::ModelError(_, _) => "model_error",
        MistralResponse::CompletionModelError(_, _) => "completion_model_error",
    }
}

fn mistral_response_error(response: &MistralResponse) -> Option<String> {
    match response {
        MistralResponse::InternalError(error) | MistralResponse::ValidationError(error) => {
            Some(error.to_string())
        }
        MistralResponse::ModelError(message, _)
        | MistralResponse::CompletionModelError(message, _) => Some(message.clone()),
        _ => None,
    }
}
