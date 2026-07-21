use crate::catalog::{ModelFileKind, ModelManifest};
use crate::engine::mistralrs::MistralRsRuntimeConfig;
use crate::engine::mistralrs::mapping::{map_multimodal_request, map_multimodal_response};
use crate::{Error, Result};
use candle_core::Device;
use futures_util::{StreamExt, stream};
use mistralrs_core::{
    AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting, LoaderBuilder, MistralRsBuilder,
    ModelDType, ModelPaths, ModelSelected, NormalRequest, PagedAttentionConfig, Request,
    RequestMessage, SamplingParams, SchedulerConfig, TokenSource, UQFF_MULTI_FILE_DELIMITER,
    get_auto_device_map_params,
};
use nexo_core::inference::requests::MultiModalPayload;
use nexo_core::{
    EmbedResponse, EmbeddingVector, InferenceMeta, InferenceOperation, InferenceOutput,
    InferenceRequest, InferenceStream, InferenceUpdate, ModelId, ModelRuntimeState, StreamSeq,
};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tracing::warn;

/// Model Runtime for the MistralRs inference engine.
pub(crate) struct MistralRsRuntime {
    /// The manifest that defines the model being loaded into this runtime instance.
    manifest: ModelManifest,

    /// The live Mistral.rs runtime once the model has been loaded.
    runtime: Option<Arc<mistralrs_core::MistralRs>>,

    /// Monotonic ordinal assigned to requests submitted to the backing runtime.
    next_request_ordinal: AtomicUsize,
}

impl MistralRsRuntime {
    /// Creates a new unloaded Mistral.rs runtime for the provided model manifest.
    ///
    /// # Arguments
    ///
    /// * `manifest` - The local manifest that provides the model definition and storage paths.
    pub(crate) fn new(manifest: ModelManifest) -> Self {
        Self {
            manifest,
            runtime: None,
            next_request_ordinal: AtomicUsize::new(1),
        }
    }

    /// Loads the model described by this runtime's manifest into Mistral.rs.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The concrete model identifier being transitioned to the loaded state.
    pub(crate) async fn load_model(&mut self, model_id: &ModelId) -> Result {
        if self.runtime.is_some() {
            warn!(model_id = %model_id, "Model is already loaded in MistralRs runtime");
            return Ok(());
        }

        let runtime_config = apply_model_specific_runtime_config(
            MistralRsRuntimeConfig::default(),
            self.manifest.model_id(),
        );
        let device = metal_device()?;
        let pipeline = build_pipeline(&self.manifest, &runtime_config, &device)?;
        let scheduler = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(NonZeroUsize::MIN),
        };

        let runtime =
            MistralRsBuilder::new(pipeline, scheduler, runtime_config.throughput_logging, None)
                .with_model_id(model_id.to_string())
                .with_no_kv_cache(runtime_config.no_kv_cache)
                .with_no_prefix_cache(runtime_config.no_prefix_cache)
                .with_prefix_cache_n(runtime_config.prefix_cache_entries)
                .with_disable_eos_stop(runtime_config.disable_eos_stop)
                .build()
                .await;

        self.runtime = Some(runtime);
        self.next_request_ordinal.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Unloads the currently loaded model from the Mistral.rs runtime.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The concrete model identifier being transitioned to the unloaded state.
    pub(crate) async fn unload_model(&mut self, model_id: &ModelId) -> Result {
        if self.runtime.take().is_none() {
            warn!(model_id = %model_id, "Model is not loaded in MistralRs runtime");
            return Ok(());
        }

        self.next_request_ordinal.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Submits an inference request to the loaded Mistral.rs runtime.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The concrete model identifier targeted by the request.
    /// * `request` - The shared inference request that should be executed by the runtime.
    pub(crate) async fn infer(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
    ) -> Result<InferenceStream> {
        if self.runtime.is_none() {
            return Err(Error::ModelNotLoaded {
                model_id: model_id.clone(),
                current_state: ModelRuntimeState::Unloaded,
            });
        }

        let meta = InferenceMeta::from_request(&request);
        match request.operation.clone() {
            InferenceOperation::MultiModal(payload) => {
                self.infer_multimodal(model_id, request, meta, payload)
                    .await
            }
            InferenceOperation::Embed(payload) => self.infer_embeddings(model_id, meta, payload),
            _ => Err(Error::UnsupportedFeature {
                feature: format!(
                    "operation `{}` is not implemented yet for model `{model_id}` on the MistralRs runtime",
                    request.operation_kind()
                ),
            }),
        }
    }
}

impl MistralRsRuntime {
    /// Executes the multimodal generation path for a loaded Mistral.rs model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The concrete model identifier targeted by the request.
    /// * `request` - The full inference request carrying operation and session identity.
    /// * `meta` - Stable execution metadata emitted in each streamed update.
    /// * `payload` - The multimodal generation payload to map and dispatch.
    async fn infer_multimodal(
        &self,
        model_id: &ModelId,
        request: InferenceRequest,
        meta: InferenceMeta,
        payload: MultiModalPayload,
    ) -> Result<InferenceStream> {
        let runtime = self.runtime.clone().ok_or_else(|| Error::ModelNotLoaded {
            model_id: model_id.clone(),
            current_state: ModelRuntimeState::Unloaded,
        })?;
        let descriptor = self.manifest.definition().clone();
        let request_ordinal = self.next_request_ordinal.fetch_add(1, Ordering::Relaxed);
        let (response_tx, response_rx) = mpsc::channel(32);
        let mistral_request = map_multimodal_request(
            &request,
            &payload,
            &descriptor,
            response_tx,
            request_ordinal,
        )?;

        dispatch_request(
            runtime.clone(),
            model_id,
            Request::Normal(Box::new(mistral_request)),
        )
        .await?;

        let started = stream::once({
            let meta = meta.clone();
            async move { Ok(InferenceUpdate::started(meta)) }
        });
        let body = stream::unfold(
            (response_rx, meta, StreamSeq::first()),
            |(mut response_rx, meta, seq)| async move {
                let response = response_rx.recv().await?;
                let update = map_multimodal_response(response, &meta, seq);
                let next_seq = if update.as_ref().is_ok_and(InferenceUpdate::is_progress) {
                    seq.next()
                } else {
                    seq
                };

                Some((update, (response_rx, meta, next_seq)))
            },
        );

        Ok(started.chain(body).boxed())
    }

    /// Executes the embedding path for a loaded Mistral.rs model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The concrete model identifier targeted by the request.
    /// * `meta` - Stable execution metadata emitted in each streamed update.
    /// * `payload` - The embedding payload containing the ordered input prompts.
    fn infer_embeddings(
        &self,
        model_id: &ModelId,
        meta: InferenceMeta,
        payload: nexo_core::EmbedPayload,
    ) -> Result<InferenceStream> {
        let runtime = self.runtime.clone().ok_or_else(|| Error::ModelNotLoaded {
            model_id: model_id.clone(),
            current_state: ModelRuntimeState::Unloaded,
        })?;
        let model_id = model_id.clone();
        let request_ordinals = (0..payload.inputs.len())
            .map(|_| self.next_request_ordinal.fetch_add(1, Ordering::Relaxed))
            .collect::<Vec<_>>();

        let started = stream::once({
            let meta = meta.clone();
            async move { Ok(InferenceUpdate::started(meta)) }
        });

        let completed = stream::once(async move {
            let mut vectors = Vec::with_capacity(payload.inputs.len());
            let mut usage = nexo_core::TokenUsage::default();

            for (index, (input, request_ordinal)) in payload
                .inputs
                .into_iter()
                .zip(request_ordinals.into_iter())
                .enumerate()
            {
                let response =
                    execute_embedding_request(runtime.clone(), &model_id, input, request_ordinal)
                        .await
                        .map_err(|error| nexo_core::Error::Inference {
                            message: error.to_string(),
                        })?;

                match response {
                    mistralrs_core::Response::Embeddings {
                        embeddings,
                        prompt_tokens,
                        total_tokens,
                    } => {
                        vectors.push(EmbeddingVector {
                            index,
                            values: embeddings,
                        });
                        usage.input_tokens += prompt_tokens;
                        usage.total_tokens += total_tokens;
                    }
                    other => {
                        return Err(nexo_core::Error::Inference {
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

            let final_output = InferenceOutput::Embed(EmbedResponse {
                vectors,
                usage: Some(usage),
            });

            Ok(InferenceUpdate::completed(meta, final_output))
        });

        Ok(started.chain(completed).boxed())
    }
}

async fn execute_embedding_request(
    runtime: Arc<mistralrs_core::MistralRs>,
    model_id: &ModelId,
    prompt: String,
    request_ordinal: usize,
) -> Result<mistralrs_core::Response> {
    let (response_tx, mut response_rx) = mpsc::channel(1);
    let mut request = NormalRequest::new_simple(
        RequestMessage::Embedding { prompt },
        SamplingParams::neutral(),
        response_tx,
        request_ordinal,
        None,
        None,
    );
    request.model_id = Some(model_id.to_string());

    dispatch_request(runtime, model_id, Request::Normal(Box::new(request))).await?;

    response_rx.recv().await.ok_or_else(|| Error::Runtime {
        message: "embedding response channel closed before producing output".to_string(),
    })
}

async fn dispatch_request(
    runtime: Arc<mistralrs_core::MistralRs>,
    model_id: &ModelId,
    request: Request,
) -> Result {
    let sender = runtime
        .get_sender(Some(model_id.as_ref()))
        .map_err(|error| Error::Runtime {
            message: error.to_string(),
        })?;

    sender.send(request).await.map_err(|_| Error::Runtime {
        message: format!("failed to dispatch request to model `{model_id}`"),
    })
}

/// Builds the loaded Mistral.rs pipeline for the supplied manifest.
///
/// # Arguments
///
/// * `manifest` - The manifest that resolves local model files and metadata.
/// * `runtime_config` - The active runtime configuration applied during model load.
/// * `device` - The target device used for pipeline construction.
fn build_pipeline(
    manifest: &ModelManifest,
    runtime_config: &MistralRsRuntimeConfig,
    device: &Device,
) -> Result<Arc<tokio::sync::Mutex<dyn mistralrs_core::Pipeline + Send + Sync>>> {
    let model_dir = manifest.model_dir()?;
    let from_uqff = resolve_uqff_selection(manifest)?;
    let selected_model_id = if from_uqff.is_some() {
        model_dir.to_string_lossy().into_owned()
    } else {
        manifest.model_id().to_string()
    };
    let selected = ModelSelected::Run {
        model_id: selected_model_id,
        tokenizer_json: None,
        dtype: ModelDType::Auto,
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
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let loader = LoaderBuilder::new(selected.clone())
        .with_no_kv_cache(runtime_config.no_kv_cache)
        .with_chat_template(path_to_string(
            manifest
                .first_local_file_by_kind(&[
                    ModelFileKind::ChatTemplate,
                    ModelFileKind::TokenizerConfig,
                ])?
                .as_ref(),
        ))
        .with_jinja_explicit(path_to_string(
            manifest
                .first_local_file_by_kind(&[ModelFileKind::ChatTemplateJson])?
                .as_ref(),
        ))
        .build()
        .map_err(|error| Error::Runtime {
            message: error.to_string(),
        })?;

    let dtype = ModelDType::Auto;
    let device_map = get_auto_device_map_params(&selected).map_err(|error| Error::Runtime {
        message: error.to_string(),
    })?;

    if selected_uses_uqff(&selected) {
        return loader
            .load_model_from_hf(
                None,
                TokenSource::None,
                &dtype,
                device,
                true,
                DeviceMapSetting::Auto(device_map),
                None,
                None::<PagedAttentionConfig>,
            )
            .map_err(|error| Error::Runtime {
                message: error.to_string(),
            });
    }

    let local_paths = build_local_model_paths(manifest)?;
    loader
        .load_model_from_path(
            &local_paths,
            &dtype,
            device,
            true,
            DeviceMapSetting::Auto(device_map),
            None,
            None::<PagedAttentionConfig>,
        )
        .map_err(|error| Error::Runtime {
            message: error.to_string(),
        })
}

/// Applies model-family-specific runtime configuration adjustments.
///
/// # Arguments
///
/// * `runtime_config` - The configured runtime options before model-specific adjustments.
/// * `model_id` - The concrete model identifier used to select compatibility tweaks.
fn apply_model_specific_runtime_config(
    runtime_config: MistralRsRuntimeConfig,
    model_id: &ModelId,
) -> MistralRsRuntimeConfig {
    if matches!(model_id.family(), nexo_core::ModelFamily::Gemma4) {
        warn!(
            "Consider applying no_prefix_cache = true to gemma4 as there used to be a bug. Might be resolved already now."
        );
        // runtime_config.no_prefix_cache = true;
    }

    runtime_config
}

/// Returns whether the selected model load path resolves through UQFF artifacts.
///
/// # Arguments
///
/// * `selected` - The Mistral.rs model selection being inspected.
fn selected_uses_uqff(selected: &ModelSelected) -> bool {
    matches!(
        selected,
        ModelSelected::Run {
            from_uqff: Some(_),
            ..
        }
    )
}

/// Builds the local model-path bundle expected by Mistral.rs from manifest storage.
///
/// # Arguments
///
/// * `manifest` - The model manifest that declares tokenizer, config, and weight files.
fn build_local_model_paths(manifest: &ModelManifest) -> Result<Box<dyn ModelPaths>> {
    let model_dir = manifest.model_dir()?;
    let tokenizer_filename = manifest
        .first_local_file_by_kind(&[ModelFileKind::Tokenizer])?
        .ok_or_else(|| missing_local_file(&model_dir, "tokenizer artifact"))?;
    let config_filename = manifest
        .first_local_file_by_kind(&[ModelFileKind::Config])?
        .ok_or_else(|| missing_local_file(&model_dir, "config artifact"))?;
    let filenames = collect_weight_files(manifest)?;

    Ok(Box::new(mistralrs_core::LocalModelPaths {
        tokenizer_filename,
        config_filename,
        template_filename: manifest.first_local_file_by_kind(&[
            ModelFileKind::ChatTemplate,
            ModelFileKind::TokenizerConfig,
        ])?,
        filenames,
        adapter_paths: mistralrs_core::AdapterPaths::None,
        gen_conf: manifest.first_local_file_by_kind(&[ModelFileKind::GenerationConfig])?,
        preprocessor_config: manifest
            .first_local_file_by_kind(&[ModelFileKind::PreprocessorConfig])?,
        processor_config: manifest.first_local_file_by_kind(&[ModelFileKind::ProcessorConfig])?,
        chat_template_json_filename: manifest
            .first_local_file_by_kind(&[ModelFileKind::ChatTemplateJson])?,
    }))
}

/// Recursively collects local model weight files from a model directory.
///
/// # Arguments
///
/// * `manifest` - The model manifest that declares model weight files.
fn collect_weight_files(manifest: &ModelManifest) -> Result<Vec<PathBuf>> {
    let model_dir = manifest.model_dir()?;
    let files = manifest.local_files_by_kind(&[
        ModelFileKind::Weights,
        ModelFileKind::WeightShard,
        ModelFileKind::UqffResidual,
    ])?;

    if files.is_empty() {
        return Err(Error::Runtime {
            message: format!(
                "no local model weight files found under `{}`; run `nexo-ai models pull <model>` first",
                model_dir.display()
            ),
        });
    }

    Ok(files)
}

/// Detects whether the local model directory contains UQFF artifacts.
///
/// # Arguments
///
/// * `manifest` - The model manifest that declares UQFF shard files.
fn resolve_uqff_selection(manifest: &ModelManifest) -> Result<Option<String>> {
    let files = manifest
        .local_files_by_kind(&[ModelFileKind::UqffShard])?
        .into_iter()
        .filter_map(|path| {
            path.file_name()
                .and_then(|filename| filename.to_str())
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();

    if files.is_empty() {
        Ok(None)
    } else {
        Ok(Some(files.join(UQFF_MULTI_FILE_DELIMITER)))
    }
}

fn missing_local_file(model_dir: &Path, filename: &str) -> Error {
    Error::Runtime {
        message: format!(
            "missing local model file `{}` under `{}`; run `nexo-ai models pull <model>` first",
            filename,
            model_dir.display()
        ),
    }
}

#[cfg(feature = "metal")]
fn metal_device() -> Result<Device> {
    Device::new_metal(0).map_err(|error| Error::Runtime {
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

fn path_to_string(path: Option<&PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().into_owned())
}
