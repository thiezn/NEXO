use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use futures_util::StreamExt;
use nexo_ai::{
    InferenceEngine, InferenceEngineConfig, RegisteredModelConfig, RuntimeConfig,
    StaticModelRegistry,
};
use nexo_core::inference::request::GenerateRequest;
use nexo_core::{
    AudioFormat, ContentPart, ConversationMessage, FinishReason, GenerateDelta, InferenceRequest,
    InferenceResponse, InferenceStream, MediaSource, MessageRole, ModelCapability, ModelDescriptor,
    ModelId, ModelSelection, PerformanceMetrics, RequestId, RoundId, RunId, SessionId, TextPart,
    TokenUsage, ToolCall, ToolCallDelta, ToolCallId, ToolChoice,
};
use nexo_ws_client::WriteHalf;
use nexo_ws_schema::{
    AudioAnalyzeParams, AudioAnalyzeResponse, AudioGenerateParams, AudioGenerateResponse,
    ErrorPayload, Frame, GeneratedImagePayload, ImageAnalyzeParams, ImageAnalyzeResponse,
    ImageGenerateParams, ImageGenerateResponse, Method, ModelLoadParams, ModelLoadResponse,
    ModelStatusParams, ModelUnloadParams, ModelUnloadResponse, RunRoundRequest, RunRoundResponse,
    RunRoundToolCall,
};
use tokio::sync::Mutex;

use crate::config::NodeConfig;

use super::protocol::send;

pub(super) type SharedModels = Arc<Mutex<LoadedModels>>;
pub(super) type InferenceResult = Result<serde_json::Value, String>;
pub(super) type InferenceSender = tokio::sync::mpsc::Sender<(String, InferenceResult)>;

#[derive(Debug)]
pub(super) struct LoadedModels {
    runtime: RuntimeConfig,
    enable_tool_calling: bool,
    available: BTreeMap<ModelId, RegisteredModelConfig>,
    engine: Option<InferenceEngine>,
}

impl LoadedModels {
    pub(super) fn new(
        runtime: RuntimeConfig,
        enable_tool_calling: bool,
        models: Vec<RegisteredModelConfig>,
    ) -> Self {
        Self {
            runtime,
            enable_tool_calling,
            available: models
                .into_iter()
                .map(|model| (model.descriptor.id.clone(), model))
                .collect(),
            engine: None,
        }
    }

    fn enable_tool_calling(&self) -> bool {
        self.enable_tool_calling
    }

    pub(super) fn available_model_ids(&self) -> Vec<String> {
        self.available.keys().map(ToString::to_string).collect()
    }

    pub(super) fn available_model_descriptors(&self) -> Vec<ModelDescriptor> {
        self.available
            .values()
            .map(|model| model.descriptor.clone())
            .collect()
    }

    async fn loaded_model_descriptors(&self) -> Vec<ModelDescriptor> {
        match &self.engine {
            Some(engine) => engine.loaded_models().await,
            None => Vec::new(),
        }
    }

    fn startup_model_ids(&self, config: &NodeConfig) -> Vec<ModelId> {
        let descriptors = self
            .available
            .values()
            .map(|model| model.descriptor.clone())
            .collect::<Vec<_>>();
        let Ok(registry) = StaticModelRegistry::new(descriptors) else {
            if !config.startup_capabilities.is_empty() {
                tracing::warn!("No downloaded models available for startup capabilities");
            }
            return Vec::new();
        };

        let mut selected = Vec::new();
        let mut selected_ids = BTreeSet::new();
        let mut satisfied = HashSet::new();

        for (index, capability) in config.startup_capabilities.iter().copied().enumerate() {
            if satisfied.contains(&capability) {
                continue;
            }

            let specific_model = config.default_models.get(&capability).cloned();
            let runtime_preference = specific_model
                .as_ref()
                .and_then(|model_id| self.available.get(model_id))
                .map(|model| model.descriptor.runtime)
                .unwrap_or(nexo_core::InferenceRuntime::AnyTts);

            let selection = ModelSelection {
                specific_model,
                required_capabilities: vec![capability],
                preferred_capabilities: config.startup_capabilities[index + 1..].to_vec(),
                runtime_preference,
            };
            let Some(descriptor) = registry.resolve_model(&selection) else {
                tracing::warn!(
                    "No downloaded model available for startup capability {capability:?}"
                );
                continue;
            };

            if selected_ids.insert(descriptor.id.clone()) {
                selected.push(descriptor.id.clone());
            }
            satisfied.extend(descriptor.capabilities.iter().copied());
        }

        selected
    }

    async fn load_model(&mut self, model_id: &str) -> Result<(), String> {
        let model_id = ModelId::from(model_id);
        if !self.available.contains_key(&model_id) {
            return Err(format!("Model '{model_id}' is not downloaded on this node"));
        }

        self.ensure_engine().await?;
        let engine = self
            .engine
            .clone()
            .ok_or_else(|| "No inference engine configured".to_string())?;
        engine
            .load_model(&model_id, self.available[&model_id].descriptor.runtime)
            .await
            .map_err(|error| error.to_string())
    }

    async fn unload_model(&mut self, model_id: &str) -> Result<bool, String> {
        let model_id = ModelId::from(model_id);
        let Some(engine) = self.engine.clone() else {
            return Ok(false);
        };

        engine
            .unload_model(&model_id)
            .await
            .map_err(|error| error.to_string())
    }

    fn engine(&self) -> Result<InferenceEngine, String> {
        self.engine
            .clone()
            .ok_or_else(|| "No model loaded for inference".to_string())
    }

    async fn ensure_engine(&mut self) -> Result<(), String> {
        if self.engine.is_some() {
            return Ok(());
        }
        if self.available.is_empty() {
            return Ok(());
        }

        self.engine = Some(
            InferenceEngine::new(InferenceEngineConfig {
                runtime: self.runtime.clone(),
                models: self.available.values().cloned().collect(),
            })
            .await
            .map_err(|error| error.to_string())?,
        );
        Ok(())
    }
}

pub(super) fn shared_models(
    runtime: RuntimeConfig,
    enable_tool_calling: bool,
    models: Vec<RegisteredModelConfig>,
) -> SharedModels {
    Arc::new(Mutex::new(LoadedModels::new(
        runtime,
        enable_tool_calling,
        models,
    )))
}

pub(super) async fn load_startup_models(models: &SharedModels, config: &NodeConfig) {
    let (available_model_ids, startup_model_ids) = {
        let models = models.lock().await;
        (
            models.available_model_ids(),
            models.startup_model_ids(config),
        )
    };

    if available_model_ids.is_empty() {
        tracing::info!("No downloaded models detected");
    } else {
        tracing::info!("Detected available models: {available_model_ids:?}");
    }

    tracing::info!(
        "Startup capabilities: {:?}, models to load: {:?}",
        config.startup_capabilities,
        startup_model_ids
    );

    for model_id in startup_model_ids {
        let result = {
            let mut models = models.lock().await;
            models.load_model(model_id.as_str()).await
        };

        match result {
            Ok(()) => tracing::info!("Auto-loaded startup model '{model_id}'"),
            Err(error) => tracing::warn!("Failed to auto-load '{model_id}': {error}"),
        }
    }
}

pub(super) async fn push_model_status(writer: &mut WriteHalf, models: &SharedModels) {
    let (loaded_models, available_models, available_model_descriptors) = {
        let models = models.lock().await;
        (
            models.loaded_model_descriptors().await,
            models.available_model_ids(),
            models.available_model_descriptors(),
        )
    };

    let status = ModelStatusParams {
        loaded_models,
        available_models,
        available_model_descriptors,
    };
    if let Ok(frame) = Frame::request(Method::ModelStatus, &status) {
        let _ = writer.send_frame(&frame).await;
    }
}

pub(super) async fn handle_model_load(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
) -> cli_helpers::Result {
    let params: ModelLoadParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            send_invalid_params_error(
                writer,
                request_id,
                format!("Invalid model.load params: {error}"),
            )
            .await?;
            return Ok(());
        }
    };

    let model_id = params.model_id.trim();
    if model_id.is_empty() {
        send_invalid_params_error(
            writer,
            request_id,
            "Parameter 'modelId' must not be empty".to_string(),
        )
        .await?;
        return Ok(());
    }

    tracing::info!("Loading model '{model_id}'");
    let result = {
        let mut models = models.lock().await;
        models.load_model(model_id).await
    };
    let (loaded, error) = match result {
        Ok(()) => {
            tracing::info!(model_id, "Model loaded");
            (true, None)
        }
        Err(error) => {
            tracing::error!(model_id, error = %error, "Failed to load model");
            (false, Some(error))
        }
    };

    let response = Frame::ok_response(
        request_id,
        &ModelLoadResponse {
            model_id: model_id.to_string(),
            loaded,
            error,
        },
    )
    .unwrap_or_else(internal_error_response(request_id));

    send(writer, &response).await?;
    push_model_status(writer, models).await;

    Ok(())
}

pub(super) async fn handle_model_unload(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
) -> cli_helpers::Result {
    let params: ModelUnloadParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            send_invalid_params_error(
                writer,
                request_id,
                format!("Invalid model.unload params: {error}"),
            )
            .await?;
            return Ok(());
        }
    };

    let model_id = params.model_id.trim();
    if model_id.is_empty() {
        send_invalid_params_error(
            writer,
            request_id,
            "Parameter 'modelId' must not be empty".to_string(),
        )
        .await?;
        return Ok(());
    }

    tracing::info!("Unloading model '{model_id}'");
    let result = {
        let mut models = models.lock().await;
        models.unload_model(model_id).await
    };

    let response = match result {
        Ok(unloaded) => Frame::ok_response(request_id, &ModelUnloadResponse { unloaded })
            .unwrap_or_else(internal_error_response(request_id)),
        Err(message) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "model_unload_failed".into(),
                message,
            },
        ),
    };

    send(writer, &response).await?;
    push_model_status(writer, models).await;

    Ok(())
}

pub(super) async fn queue_run_round(
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let request: RunRoundRequest = match serde_json::from_value(params) {
        Ok(request) => request,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid typed run round params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    let enable_tool_calling = {
        let models = models.lock().await;
        models.enable_tool_calling()
    };
    tracing::info!(
        request_id,
        run_id = %request.run_id,
        round_id = %request.round_id,
        session_id = %request.session_id,
        model_id = ?request.model_id,
        messages = request.messages.len(),
        tools = request.tools.len(),
        has_tools = !request.tools.is_empty(),
        tool_calling_enabled = enable_tool_calling,
        tool_choice = ?request.tool_choice,
        reasoning = ?request.reasoning,
        "Starting round inference"
    );

    let runtime_preference = {
        let models = models.lock().await;
        request
            .model_id
            .as_ref()
            .map(|model_id| ModelId::from(model_id.as_str()))
            .and_then(|model_id| models.available.get(&model_id))
            .map(|model| model.descriptor.runtime)
            .unwrap_or(nexo_core::InferenceRuntime::AnyTts)
    };

    let request = run_round_request(request_id, request, enable_tool_calling, runtime_preference);
    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_run_round(&models, request).await;
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

pub(super) async fn queue_image_analyze(
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let params: ImageAnalyzeParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid image.analyze params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    tracing::info!(
        session_id = ?params.session_id,
        media_type = ?params.media_type,
        image_base64_chars = params.image_data.len(),
        image_bytes_estimate = estimate_base64_bytes(params.image_data.len()),
        max_tokens = params.max_tokens,
        temperature = params.temperature,
        "Analyzing image (prompt: '{:.80}')",
        params.prompt
    );
    let mut request = InferenceRequest::Generate(GenerateRequest::new_image_analyze(
        RequestId::from(request_id),
        params.image_data,
        params.media_type,
        params.prompt,
        params.max_tokens,
        params.temperature as f32,
    ));
    if let InferenceRequest::Generate(generate) = &mut request {
        generate.session_id = params.session_id.map(SessionId::from);
    }
    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_image_analyze(&models, request).await;
        match &result {
            Ok(_) => tracing::info!(request_id, "Image analyze inference completed"),
            Err(error) => {
                tracing::error!(request_id, error = %error, "Image analyze inference failed")
            }
        }
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

pub(super) async fn queue_audio_analyze(
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let params: AudioAnalyzeParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid audio.analyze params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    tracing::info!(
        session_id = ?params.session_id,
        media_type = ?params.media_type,
        audio_base64_chars = params.audio_data.len(),
        audio_bytes_estimate = estimate_base64_bytes(params.audio_data.len()),
        sample_rate_hz = ?params.sample_rate_hz,
        channel_count = ?params.channel_count,
        max_tokens = params.max_tokens,
        temperature = params.temperature,
        "Analyzing audio (prompt: '{:.80}')",
        params.prompt
    );
    let mut request = InferenceRequest::Generate(GenerateRequest::new_audio_analyze(
        RequestId::from(request_id),
        params.audio_data,
        params.media_type,
        params.sample_rate_hz,
        params.channel_count,
        params.prompt,
        params.max_tokens,
        params.temperature as f32,
    ));
    if let InferenceRequest::Generate(generate) = &mut request {
        generate.session_id = params.session_id.map(SessionId::from);
    }
    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_audio_analyze(&models, request).await;
        match &result {
            Ok(_) => tracing::info!(request_id, "Audio analyze inference completed"),
            Err(error) => {
                tracing::error!(request_id, error = %error, "Audio analyze inference failed")
            }
        }
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

pub(super) async fn queue_image_generate(
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let params: ImageGenerateParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid image.generate params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    tracing::info!(
        session_id = ?params.session_id,
        width = params.width,
        height = params.height,
        sample_count = params.sample_count,
        steps = ?params.steps,
        guidance_scale = ?params.guidance_scale,
        seed = ?params.seed,
        "Generating image(s) (prompt: '{:.80}')",
        params.prompt
    );

    let request = InferenceRequest::GenerateImage(nexo_core::ImageGenerationRequest {
        request_id: Some(RequestId::from(request_id)),
        session_id: params.session_id.map(SessionId::from),
        model: ModelSelection {
            specific_model: None,
            required_capabilities: vec![ModelCapability::ImageGeneration],
            preferred_capabilities: Vec::new(),
            runtime_preference: nexo_core::InferenceRuntime::AnyTts,
        },
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        size: nexo_core::ImageGenerationSize {
            width: params.width,
            height: params.height,
        },
        sample_count: params.sample_count,
        steps: params.steps,
        guidance_scale: params.guidance_scale,
        seed: params.seed,
        metadata: nexo_core::MetadataMap::new(),
    });

    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_image_generate(&models, request).await;
        match &result {
            Ok(_) => tracing::info!(request_id, "Image generation inference completed"),
            Err(error) => {
                tracing::error!(request_id, error = %error, "Image generation inference failed")
            }
        }
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

pub(super) async fn queue_audio_generate(
    request_id: &str,
    params: serde_json::Value,
    models: &SharedModels,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let params: AudioGenerateParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid audio.generate params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    tracing::info!(
        session_id = ?params.session_id,
        voice = ?params.voice,
        sample_rate_hz = ?params.sample_rate_hz,
        speed = ?params.speed,
        "Generating audio (prompt: '{:.80}')",
        params.prompt
    );

    let request = InferenceRequest::GenerateSpeech(nexo_core::SpeechGenerationRequest {
        request_id: Some(RequestId::from(request_id)),
        session_id: params.session_id.map(SessionId::from),
        model: ModelSelection {
            specific_model: None,
            required_capabilities: vec![ModelCapability::SpeechGeneration],
            preferred_capabilities: Vec::new(),
            runtime_preference: nexo_core::InferenceRuntime::AnyTts,
        },
        text: params.prompt,
        language: params.language,
        voice: params.voice,
        format: AudioFormat::Wav,
        sample_rate_hz: params.sample_rate_hz,
        speed: params.speed,
        metadata: nexo_core::MetadataMap::new(),
    });

    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_audio_generate(&models, request).await;
        match &result {
            Ok(_) => tracing::info!(request_id, "Audio generation inference completed"),
            Err(error) => {
                tracing::error!(request_id, error = %error, "Audio generation inference failed")
            }
        }
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

async fn execute_run_round(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let trace_label = request_trace_label(&request);
    let stream = submit(models, request).await?;
    let response = run_round_response_from_stream(stream, &trace_label).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn execute_image_analyze(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let trace_label = request_trace_label(&request);
    let stream = submit(models, request).await?;
    let response = image_analyze_response_from_stream(stream, &trace_label).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn execute_audio_analyze(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let trace_label = request_trace_label(&request);
    let stream = submit(models, request).await?;
    let response = audio_analyze_response_from_stream(stream, &trace_label).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn execute_image_generate(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let trace_label = request_trace_label(&request);
    let stream = submit(models, request).await?;
    let response = image_generate_response_from_stream(stream, &trace_label).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn execute_audio_generate(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let trace_label = request_trace_label(&request);
    let stream = submit(models, request).await?;
    let response = audio_generate_response_from_stream(stream, &trace_label).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn submit(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<InferenceStream, String> {
    let engine = {
        let models = models.lock().await;
        models.engine()?
    };

    engine
        .submit(request)
        .await
        .map_err(|error| error.to_string())
}

fn run_round_request(
    request_id: &str,
    round: RunRoundRequest,
    enable_tool_calling: bool,
    runtime_preference: nexo_core::InferenceRuntime,
) -> InferenceRequest {
    let use_tools = enable_tool_calling
        && !round.tools.is_empty()
        && !matches!(round.tool_choice, ToolChoice::Disabled);
    let tool_choice = if use_tools {
        round.tool_choice
    } else {
        ToolChoice::Disabled
    };
    InferenceRequest::Generate(GenerateRequest::new_round(
        RequestId::from(request_id),
        SessionId::from(round.session_id),
        RunId::from(round.run_id),
        RoundId::from(round.round_id),
        ModelSelection {
            specific_model: round.model_id.map(ModelId::from),
            required_capabilities: vec![ModelCapability::TextGeneration],
            preferred_capabilities: if use_tools {
                vec![ModelCapability::ToolCalling]
            } else {
                Vec::new()
            },
            runtime_preference,
        },
        round.messages,
        if use_tools { round.tools } else { Vec::new() },
        tool_choice,
        round.reasoning,
    ))
}

async fn run_round_response_from_stream(
    stream: InferenceStream,
    trace_label: &str,
) -> Result<RunRoundResponse, String> {
    let output = collect_generation(stream, trace_label).await?;
    log_round_completion(trace_label, &output);
    Ok(RunRoundResponse {
        content: non_empty(output.content),
        rationale: non_empty(output.reasoning),
        tool_calls: output
            .tool_calls
            .into_iter()
            .map(|call| RunRoundToolCall { call })
            .collect(),
    })
}

async fn image_analyze_response_from_stream(
    stream: InferenceStream,
    trace_label: &str,
) -> Result<ImageAnalyzeResponse, String> {
    let output = collect_generation(stream, trace_label).await?;
    Ok(ImageAnalyzeResponse {
        text: output.content,
        tokens_generated: output.usage.map_or(0, |usage| usage.output_tokens),
        inference_time_ms: output
            .performance
            .map_or(0, |performance| performance.total_duration_ms),
    })
}

async fn audio_analyze_response_from_stream(
    stream: InferenceStream,
    trace_label: &str,
) -> Result<AudioAnalyzeResponse, String> {
    let output = collect_generation(stream, trace_label).await?;
    Ok(AudioAnalyzeResponse {
        text: output.content,
        tokens_generated: output.usage.map_or(0, |usage| usage.output_tokens),
        inference_time_ms: output
            .performance
            .map_or(0, |performance| performance.total_duration_ms),
    })
}

async fn image_generate_response_from_stream(
    mut stream: InferenceStream,
    trace_label: &str,
) -> Result<ImageGenerateResponse, String> {
    while let Some(response) = stream.next().await {
        let response = response.map_err(|error| error.to_string())?;
        match response {
            InferenceResponse::Images(payload) => {
                let images = payload
                    .images
                    .into_iter()
                    .map(map_generated_image_payload)
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(ImageGenerateResponse {
                    images,
                    inference_time_ms: 0,
                });
            }
            InferenceResponse::Failure(failure) => return Err(failure.message),
            other => {
                tracing::debug!(
                    trace = %trace_label,
                    response_kind = inference_response_kind(&other),
                    "Ignoring non-image-generation response while waiting for image output"
                );
            }
        }
    }
    Err("Image generation stream ended without image output".to_string())
}

async fn audio_generate_response_from_stream(
    mut stream: InferenceStream,
    trace_label: &str,
) -> Result<AudioGenerateResponse, String> {
    while let Some(response) = stream.next().await {
        let response = response.map_err(|error| error.to_string())?;
        match response {
            InferenceResponse::Speech(payload) => {
                let audio_data = media_source_to_base64(payload.audio.source)?;
                let format = audio_format_name(payload.audio.format);
                let media_type = match payload.audio.format {
                    AudioFormat::Pcm => Some("audio/L16".to_string()),
                    AudioFormat::Wav => Some("audio/wav".to_string()),
                    AudioFormat::Mp3 => Some("audio/mpeg".to_string()),
                };

                return Ok(AudioGenerateResponse {
                    audio_data,
                    media_type,
                    format,
                    sample_rate_hz: payload.audio.sample_rate_hz,
                    channel_count: payload.audio.channel_count,
                    inference_time_ms: 0,
                });
            }
            InferenceResponse::Failure(failure) => return Err(failure.message),
            other => {
                tracing::debug!(
                    trace = %trace_label,
                    response_kind = inference_response_kind(&other),
                    "Ignoring non-speech-generation response while waiting for audio output"
                );
            }
        }
    }
    Err("Audio generation stream ended without audio output".to_string())
}

fn map_generated_image_payload(
    image: nexo_core::inference::request::GeneratedImage,
) -> Result<GeneratedImagePayload, String> {
    Ok(GeneratedImagePayload {
        index: image.index,
        image_data: media_source_to_base64(image.source)?,
        media_type: image.media_type,
        width: image.width,
        height: image.height,
    })
}

fn media_source_to_base64(source: MediaSource) -> Result<String, String> {
    match source {
        MediaSource::Base64(encoded) => Ok(encoded),
        MediaSource::Bytes(bytes) => {
            use base64::Engine;
            Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        MediaSource::Url(url) => Err(format!(
            "Generation returned URL media source '{url}', expected inline content"
        )),
    }
}

fn audio_format_name(format: AudioFormat) -> String {
    match format {
        AudioFormat::Pcm => "pcm",
        AudioFormat::Wav => "wav",
        AudioFormat::Mp3 => "mp3",
    }
    .to_string()
}

#[derive(Default)]
struct GenerationOutput {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: Option<FinishReason>,
    usage: Option<TokenUsage>,
    performance: Option<PerformanceMetrics>,
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<ToolCallId>,
    name: Option<String>,
    arguments: String,
}

async fn collect_generation(
    mut stream: InferenceStream,
    trace_label: &str,
) -> Result<GenerationOutput, String> {
    let mut output = GenerationOutput::default();
    let mut partial_tool_calls = BTreeMap::<usize, PartialToolCall>::new();
    let mut failure: Option<String> = None;

    while let Some(response) = stream.next().await {
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                if failure.is_none() {
                    tracing::warn!(
                        trace = %trace_label,
                        error = %message,
                        partial_content_chars = output.content.chars().count(),
                        partial_reasoning_chars = output.reasoning.chars().count(),
                        partial_tool_calls = output.tool_calls.len().max(partial_tool_calls.len()),
                        partial_content_preview = preview_text(&output.content),
                        partial_reasoning_preview = preview_text(&output.reasoning),
                        "Generation stream item failed; draining response channel before returning error"
                    );
                    failure = Some(message);
                } else {
                    tracing::warn!(trace = %trace_label, error = %message, "Additional generation stream item failed after first failure");
                }
                continue;
            }
        };

        if failure.is_some() {
            tracing::debug!(
                trace = %trace_label,
                response_kind = inference_response_kind(&response),
                "Drained late generation response after failure"
            );
            continue;
        }

        match response {
            InferenceResponse::GenerationStarted(_) => {}
            InferenceResponse::GenerationChunk(chunk) => {
                if let Some(finish_reason) = chunk.finish_reason {
                    output.finish_reason = Some(finish_reason);
                }
                apply_delta(&mut output, &mut partial_tool_calls, chunk.delta);
                if let Some(usage) = chunk.usage {
                    output.usage = Some(usage);
                }
            }
            InferenceResponse::GenerationCompleted(completed) => {
                let content = text_from_message(&completed.message);
                if !content.is_empty() {
                    output.content = content;
                }
                if let Some(reasoning) = completed.reasoning
                    && !reasoning.is_empty()
                {
                    output.reasoning = reasoning;
                }
                let tool_calls = tool_calls_from_message(completed.message);
                if !tool_calls.is_empty() {
                    output.tool_calls = tool_calls;
                }
                output.finish_reason = Some(completed.finish_reason);
                output.usage = completed.usage;
                output.performance = completed.performance;
            }
            InferenceResponse::Failure(inference_failure) => {
                tracing::warn!(
                    trace = %trace_label,
                    request_id = ?inference_failure.request_id,
                    run_id = ?inference_failure.run_id,
                    round_id = ?inference_failure.round_id,
                    code = ?inference_failure.code,
                    retryability = ?inference_failure.retryability,
                    error = %inference_failure.message,
                    partial_content_chars = output.content.chars().count(),
                    partial_reasoning_chars = output.reasoning.chars().count(),
                    partial_tool_calls = output.tool_calls.len().max(partial_tool_calls.len()),
                    partial_content_preview = preview_text(&output.content),
                    partial_reasoning_preview = preview_text(&output.reasoning),
                    "Generation failed; draining response channel before returning error"
                );
                failure = Some(inference_failure.message);
            }
            other => return Err(format!("Unsupported inference response: {other:?}")),
        }
    }

    if let Some(message) = failure {
        return Err(message);
    }

    if output.tool_calls.is_empty() && !partial_tool_calls.is_empty() {
        output.tool_calls = partial_tool_calls
            .into_iter()
            .filter_map(|(index, partial)| partial.into_call(index))
            .collect();
    }

    Ok(output)
}

fn log_round_completion(trace_label: &str, output: &GenerationOutput) {
    let usage = output.usage.as_ref();
    let performance = output.performance.as_ref();
    tracing::info!(
        trace = %trace_label,
        content_chars = output.content.chars().count(),
        reasoning_chars = output.reasoning.chars().count(),
        tool_calls = output.tool_calls.len(),
        input_tokens = usage.map(|usage| usage.input_tokens).unwrap_or(0),
        output_tokens = usage.map(|usage| usage.output_tokens).unwrap_or(0),
        total_tokens = usage.map(|usage| usage.total_tokens).unwrap_or(0),
        total_duration_ms = performance.map(|performance| performance.total_duration_ms).unwrap_or(0),
        input_tokens_per_second = performance.and_then(|performance| performance.input_tokens_per_second).unwrap_or(0.0),
        output_tokens_per_second = performance.and_then(|performance| performance.output_tokens_per_second).unwrap_or(0.0),
        finish_reason = ?output.finish_reason,
        "Completed round inference"
    );
    tracing::info!(
        trace = %trace_label,
        assistant_reasoning = %output.reasoning,
        "Assistant reasoning"
    );
    tracing::info!(
        trace = %trace_label,
        assistant_answer = %output.content,
        "Assistant answer"
    );
}

fn preview_text(value: &str) -> &str {
    const MAX_PREVIEW_BYTES: usize = 200;
    if value.len() <= MAX_PREVIEW_BYTES {
        return value;
    }

    let mut end = MAX_PREVIEW_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn apply_delta(
    output: &mut GenerationOutput,
    partial_tool_calls: &mut BTreeMap<usize, PartialToolCall>,
    delta: GenerateDelta,
) {
    if let Some(content) = delta.content_delta {
        output.content.push_str(&content);
    }
    if let Some(reasoning) = delta.reasoning_delta {
        output.reasoning.push_str(&reasoning);
    }
    for tool_delta in delta.tool_call_deltas {
        apply_tool_delta(partial_tool_calls, tool_delta);
    }
}

fn apply_tool_delta(
    partial_tool_calls: &mut BTreeMap<usize, PartialToolCall>,
    delta: ToolCallDelta,
) {
    let partial = partial_tool_calls.entry(delta.index).or_default();
    if let Some(id) = delta.id {
        partial.id = Some(id);
    }
    if let Some(name) = delta.name {
        partial.name = Some(name);
    }
    if let Some(arguments) = delta.arguments_delta {
        partial.arguments.push_str(&arguments);
    }
}

impl PartialToolCall {
    fn into_call(self, index: usize) -> Option<ToolCall> {
        let name = self.name?;
        let arguments = serde_json::from_str(&self.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(self.arguments));

        Some(ToolCall {
            id: self
                .id
                .unwrap_or_else(|| ToolCallId::from(format!("tool-call-{index}"))),
            index,
            name,
            arguments,
        })
    }
}

fn text_from_message(message: &ConversationMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(TextPart { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_calls_from_message(message: ConversationMessage) -> Vec<ToolCall> {
    if message.role != MessageRole::Assistant {
        return Vec::new();
    }

    message
        .parts
        .into_iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn inference_response_kind(response: &InferenceResponse) -> &'static str {
    match response {
        InferenceResponse::GenerationStarted(_) => "generation_started",
        InferenceResponse::GenerationChunk(_) => "generation_chunk",
        InferenceResponse::GenerationCompleted(_) => "generation_completed",
        InferenceResponse::Failure(_) => "failure",
        InferenceResponse::Embeddings(_) => "embeddings",
        InferenceResponse::Tokenization(_) => "tokenization",
        InferenceResponse::Detokenization(_) => "detokenization",
        InferenceResponse::Images(_) => "images",
        InferenceResponse::Speech(_) => "speech",
    }
}

fn request_trace_label(request: &InferenceRequest) -> String {
    match request {
        InferenceRequest::Generate(generate) => format!(
            "request_id={} run_id={} round_id={} session_id={} model_id={}",
            optional_id(generate.request_id.as_ref()),
            optional_id(generate.run_id.as_ref()),
            optional_id(generate.round_id.as_ref()),
            optional_id(generate.session_id.as_ref()),
            optional_id(generate.model.specific_model.as_ref())
        ),
        InferenceRequest::Embed(request) => {
            format!(
                "request_id={} kind=embed",
                optional_id(request.request_id.as_ref())
            )
        }
        InferenceRequest::Tokenize(request) => format!(
            "request_id={} kind=tokenize",
            optional_id(request.request_id.as_ref())
        ),
        InferenceRequest::Detokenize(request) => format!(
            "request_id={} kind=detokenize",
            optional_id(request.request_id.as_ref())
        ),
        InferenceRequest::GenerateImage(request) => format!(
            "request_id={} kind=generate_image",
            optional_id(request.request_id.as_ref())
        ),
        InferenceRequest::GenerateSpeech(request) => format!(
            "request_id={} kind=generate_speech",
            optional_id(request.request_id.as_ref())
        ),
    }
}

fn optional_id<T: ToString>(value: Option<&T>) -> String {
    value
        .map(ToString::to_string)
        .unwrap_or_else(|| "-".to_string())
}

fn estimate_base64_bytes(encoded_len: usize) -> usize {
    encoded_len.saturating_mul(3) / 4
}

async fn send_invalid_params_error(
    writer: &mut WriteHalf,
    request_id: &str,
    message: String,
) -> cli_helpers::Result {
    let error_response = Frame::error_response(
        request_id,
        ErrorPayload {
            code: "invalid_params".into(),
            message,
        },
    );

    send(writer, &error_response).await
}

fn internal_error_response<E: ToString>(request_id: &str) -> impl FnOnce(E) -> Frame + '_ {
    move |error| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: error.to_string(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use futures_util::stream;
    use nexo_ai::engine::mistralrs::{MistralRsAutoLoader, MistralRsLoader, MistralRsModelConfig};
    use nexo_ai::{ModelDataType, ModelRuntimeImplementation};
    use nexo_core::{
        GenerateChunk, InferenceErrorCode, InferenceFailure, MetadataMap, ModelModalities,
        ReasoningSettings, Retryability, RoleStrategy, SupportedModality,
    };

    use super::*;

    #[test]
    fn run_round_with_tools_prefers_tool_calling() {
        let request = run_round_request(
            "request-1",
            RunRoundRequest {
                run_id: "run-1".to_string(),
                round_id: "round-1".to_string(),
                session_id: "session-1".to_string(),
                messages: Vec::new(),
                tools: vec![nexo_core::ToolDefinition {
                    name: "ping".to_string(),
                    description: "Ping".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: None,
                    execution: nexo_core::ToolExecutionConstraints::default(),
                    metadata: MetadataMap::new(),
                }],
                tool_choice: ToolChoice::Automatic,
                reasoning: ReasoningSettings::default(),
                model_id: None,
            },
            true,
            nexo_core::InferenceRuntime::AnyTts,
        );

        let InferenceRequest::Generate(request) = request else {
            panic!("expected generate request");
        };
        assert_eq!(request.tool_choice, ToolChoice::Automatic);
        assert!(
            request
                .model
                .preferred_capabilities
                .contains(&ModelCapability::ToolCalling)
        );
    }

    #[test]
    fn run_round_with_tool_calling_disabled_omits_tools() {
        let request = run_round_request(
            "request-1",
            RunRoundRequest {
                run_id: "run-1".to_string(),
                round_id: "round-1".to_string(),
                session_id: "session-1".to_string(),
                messages: Vec::new(),
                tools: vec![nexo_core::ToolDefinition {
                    name: "ping".to_string(),
                    description: "Ping".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: None,
                    execution: nexo_core::ToolExecutionConstraints::default(),
                    metadata: MetadataMap::new(),
                }],
                tool_choice: ToolChoice::Automatic,
                reasoning: ReasoningSettings::default(),
                model_id: None,
            },
            false,
            nexo_core::InferenceRuntime::AnyTts,
        );

        let InferenceRequest::Generate(request) = request else {
            panic!("expected generate request");
        };
        assert_eq!(request.tool_choice, ToolChoice::Disabled);
        assert!(request.tools.is_empty());
        assert!(request.model.preferred_capabilities.is_empty());
    }

    #[test]
    fn run_round_with_disabled_tool_choice_omits_tools() {
        let request = run_round_request(
            "request-1",
            RunRoundRequest {
                run_id: "run-1".to_string(),
                round_id: "round-1".to_string(),
                session_id: "session-1".to_string(),
                messages: Vec::new(),
                tools: vec![nexo_core::ToolDefinition {
                    name: "ping".to_string(),
                    description: "Ping".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: None,
                    execution: nexo_core::ToolExecutionConstraints::default(),
                    metadata: MetadataMap::new(),
                }],
                tool_choice: ToolChoice::Disabled,
                reasoning: ReasoningSettings::default(),
                model_id: None,
            },
            true,
            nexo_core::InferenceRuntime::AnyTts,
        );

        let InferenceRequest::Generate(request) = request else {
            panic!("expected generate request");
        };
        assert_eq!(request.tool_choice, ToolChoice::Disabled);
        assert!(request.tools.is_empty());
        assert!(request.model.preferred_capabilities.is_empty());
    }

    #[test]
    fn startup_selection_deduplicates_model_capabilities() {
        let models = LoadedModels::new(
            RuntimeConfig::default(),
            true,
            vec![model_config(
                "multi",
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                ],
            )],
        );
        let config = NodeConfig {
            startup_capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
                ModelCapability::ImageInput,
            ],
            ..Default::default()
        };

        assert_eq!(
            models.startup_model_ids(&config),
            vec![ModelId::from("multi")]
        );
    }

    #[test]
    fn startup_selection_uses_core_model_id_defaults() {
        let models = LoadedModels::new(
            RuntimeConfig::default(),
            true,
            vec![
                model_config("first", vec![ModelCapability::TextGeneration]),
                model_config("second", vec![ModelCapability::TextGeneration]),
            ],
        );
        let mut config = NodeConfig {
            startup_capabilities: vec![ModelCapability::TextGeneration],
            ..Default::default()
        };
        config
            .default_models
            .insert(ModelCapability::TextGeneration, ModelId::from("second"));

        assert_eq!(
            models.startup_model_ids(&config),
            vec![ModelId::from("second")]
        );
    }

    #[tokio::test]
    async fn collect_generation_drains_late_responses_after_failure() {
        let stream: InferenceStream = Box::pin(stream::iter(vec![
            Ok(InferenceResponse::Failure(InferenceFailure {
                request_id: Some(RequestId::from("request-1")),
                run_id: Some(RunId::from("run-1")),
                round_id: Some(RoundId::from("round-1")),
                code: InferenceErrorCode::Internal,
                message: "Invalid sampling probability at index 0: NaN".to_string(),
                retryability: Retryability::Retryable,
            })),
            Ok(InferenceResponse::GenerationChunk(GenerateChunk {
                request_id: Some(RequestId::from("request-1")),
                run_id: Some(RunId::from("run-1")),
                round_id: Some(RoundId::from("round-1")),
                model_id: None,
                delta: GenerateDelta {
                    content_delta: Some("late".to_string()),
                    ..GenerateDelta::default()
                },
                usage: None,
                finish_reason: None,
            })),
        ]));

        let error = match collect_generation(stream, "test-request").await {
            Ok(_) => panic!("expected generation failure"),
            Err(error) => error,
        };

        assert_eq!(error, "Invalid sampling probability at index 0: NaN");
    }

    #[tokio::test]
    async fn collect_generation_returns_error_after_partial_output() {
        let stream: InferenceStream = Box::pin(stream::iter(vec![
            Ok(InferenceResponse::GenerationChunk(GenerateChunk {
                request_id: Some(RequestId::from("request-1")),
                run_id: Some(RunId::from("run-1")),
                round_id: Some(RoundId::from("round-1")),
                model_id: None,
                delta: GenerateDelta {
                    content_delta: Some("partial output".to_string()),
                    ..GenerateDelta::default()
                },
                usage: Some(TokenUsage {
                    input_tokens: 8,
                    output_tokens: 2,
                    total_tokens: 10,
                }),
                finish_reason: None,
            })),
            Ok(InferenceResponse::Failure(InferenceFailure {
                request_id: Some(RequestId::from("request-1")),
                run_id: Some(RunId::from("run-1")),
                round_id: Some(RoundId::from("round-1")),
                code: InferenceErrorCode::Internal,
                message: "runtime blew up".to_string(),
                retryability: Retryability::Retryable,
            })),
        ]));

        let error = match collect_generation(stream, "test-request").await {
            Ok(_) => panic!("expected generation failure"),
            Err(error) => error,
        };

        assert_eq!(error, "runtime blew up");
    }

    fn model_config(id: &str, capabilities: Vec<ModelCapability>) -> RegisteredModelConfig {
        RegisteredModelConfig {
            descriptor: ModelDescriptor {
                id: ModelId::from(id),
                display_name: id.to_string(),
                provider: Some("test".to_string()),
                runtime: nexo_core::InferenceRuntime::AnyTts,
                capabilities,
                modalities: ModelModalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
                role_strategy: RoleStrategy::Default,
                context_window_tokens: Some(4096),
                max_output_tokens: Some(1024),
                metadata: MetadataMap::new(),
            },
            runtimes: vec![ModelRuntimeImplementation::MistralRs(
                MistralRsModelConfig {
                    loader: MistralRsLoader::Auto(MistralRsAutoLoader {
                        model_id: id.to_string(),
                        from_uqff: None,
                        tokenizer_json: None,
                        chat_template: None,
                        jinja_explicit: None,
                        dtype: ModelDataType::Auto,
                        hf_cache_path: None,
                    }),
                    revision: None,
                },
            )],
        }
    }
}
