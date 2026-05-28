use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use futures_util::StreamExt;
use nexo_ai::{NexoAi, NexoAiConfig, RegisteredModelConfig, RuntimeConfig, StaticModelRegistry};
use nexo_core::inference::request::GenerateRequest;
use nexo_core::{
    ContentPart, Conversation, ConversationMessage, GenerateDelta, ImageInput, InferenceEngine,
    InferenceRequest, InferenceResponse, InferenceStream, MediaSource, MessageRole, MetadataMap,
    ModelCapability, ModelDescriptor, ModelId, ModelRegistry, ModelSelection, OutputConstraint,
    PerformanceMetrics, ReasoningSettings, RequestId, RoundId, RunId, SamplingConfig, SessionId,
    StreamingMode, TextPart, ThinkingMode, TokenUsage, ToolCall, ToolCallDelta, ToolCallId,
    ToolChoice,
};
use nexo_ws_client::WriteHalf;
use nexo_ws_schema::{
    ErrorPayload, Frame, ImageAnalyzeParams, ImageAnalyzeResponse, Method, ModelLoadParams,
    ModelLoadResponse, ModelStatusParams, ModelUnloadParams, ModelUnloadResponse, RunRoundRequest,
    RunRoundResponse, RunRoundToolCall,
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
    available: BTreeMap<ModelId, RegisteredModelConfig>,
    loaded: BTreeSet<ModelId>,
    engine: Option<NexoAi>,
}

impl LoadedModels {
    pub(super) fn new(runtime: RuntimeConfig, models: Vec<RegisteredModelConfig>) -> Self {
        Self {
            runtime,
            available: models
                .into_iter()
                .map(|model| (model.descriptor.id.clone(), model))
                .collect(),
            loaded: BTreeSet::new(),
            engine: None,
        }
    }

    pub(super) fn available_model_ids(&self) -> Vec<String> {
        self.available.keys().map(ToString::to_string).collect()
    }

    fn loaded_model_descriptors(&self) -> Vec<ModelDescriptor> {
        self.engine
            .as_ref()
            .map(ModelRegistry::list_models)
            .unwrap_or_default()
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

            let selection = ModelSelection {
                specific_model: config.default_models.get(&capability).cloned(),
                required_capabilities: vec![capability],
                preferred_capabilities: config.startup_capabilities[index + 1..].to_vec(),
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

        if self.loaded.contains(&model_id) {
            return Ok(());
        }

        let previous_loaded = self.loaded.clone();
        let previous_engine = self.engine.clone();
        self.loaded.insert(model_id);

        if let Err(error) = self.rebuild_engine().await {
            self.loaded = previous_loaded;
            self.engine = previous_engine;
            return Err(error);
        }

        Ok(())
    }

    async fn unload_model(&mut self, model_id: &str) -> Result<bool, String> {
        let model_id = ModelId::from(model_id);
        if !self.loaded.contains(&model_id) {
            return Ok(false);
        }

        let previous_loaded = self.loaded.clone();
        let previous_engine = self.engine.clone();
        self.loaded.remove(&model_id);

        if let Err(error) = self.rebuild_engine().await {
            self.loaded = previous_loaded;
            self.engine = previous_engine;
            return Err(error);
        }

        Ok(true)
    }

    fn engine(&self) -> Result<NexoAi, String> {
        self.engine
            .clone()
            .ok_or_else(|| "No model loaded for inference".to_string())
    }

    async fn rebuild_engine(&mut self) -> Result<(), String> {
        if self.loaded.is_empty() {
            self.engine = None;
            return Ok(());
        }

        let models = self
            .loaded
            .iter()
            .filter_map(|model_id| self.available.get(model_id).cloned())
            .collect::<Vec<_>>();
        self.engine = Some(
            NexoAi::from_config(NexoAiConfig {
                runtime: self.runtime.clone(),
                models,
            })
            .await
            .map_err(|error| error.to_string())?,
        );
        Ok(())
    }
}

pub(super) fn shared_models(
    runtime: RuntimeConfig,
    models: Vec<RegisteredModelConfig>,
) -> SharedModels {
    Arc::new(Mutex::new(LoadedModels::new(runtime, models)))
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
    let (loaded_models, available_models) = {
        let models = models.lock().await;
        (
            models.loaded_model_descriptors(),
            models.available_model_ids(),
        )
    };

    let status = ModelStatusParams {
        loaded_models,
        available_models,
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

    let response = Frame::ok_response(
        request_id,
        &ModelLoadResponse {
            model_id: model_id.to_string(),
            loaded: result.is_ok(),
            error: result.err(),
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

    tracing::debug!(
        "queue_run_round {}: model_id={:?}, has_tools={}, session_id={}, messages={}",
        request_id,
        request.model_id,
        !request.tools.is_empty(),
        request.session_id,
        request.messages.len(),
    );

    let request = run_round_request(request_id, request);
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

    tracing::info!("Analyzing image (prompt: '{:.80}')", params.prompt);
    let request = image_analyze_request(request_id, params);
    let models = models.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::spawn(async move {
        let result = execute_image_analyze(&models, request).await;
        let _ = tx.send((request_id, result)).await;
    });

    Ok(())
}

async fn execute_run_round(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let stream = submit(models, request).await?;
    let response = run_round_response_from_stream(stream).await?;
    serde_json::to_value(response).map_err(|error| error.to_string())
}

async fn execute_image_analyze(
    models: &SharedModels,
    request: InferenceRequest,
) -> Result<serde_json::Value, String> {
    let stream = submit(models, request).await?;
    let response = image_analyze_response_from_stream(stream).await?;
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

    tokio::task::spawn_blocking(move || engine.submit(request).map_err(|error| error.to_string()))
        .await
        .map_err(|error| format!("Inference task failed: {error}"))?
}

fn run_round_request(request_id: &str, round: RunRoundRequest) -> InferenceRequest {
    let has_tools = !round.tools.is_empty();
    InferenceRequest::Generate(GenerateRequest {
        request_id: Some(RequestId::from(request_id)),
        session_id: Some(SessionId::from(round.session_id)),
        run_id: Some(RunId::from(round.run_id)),
        round_id: Some(RoundId::from(round.round_id)),
        model: ModelSelection {
            specific_model: round.model_id.map(ModelId::from),
            required_capabilities: vec![ModelCapability::TextGeneration],
            preferred_capabilities: if has_tools {
                vec![ModelCapability::ToolCalling]
            } else {
                Vec::new()
            },
        },
        conversation: Conversation {
            messages: round.messages,
            metadata: MetadataMap::new(),
        },
        tools: round.tools,
        tool_choice: if has_tools {
            ToolChoice::Automatic
        } else {
            ToolChoice::Disabled
        },
        reasoning: round.reasoning,
        output_constraint: OutputConstraint::None,
        sampling: SamplingConfig::default(),
        streaming: StreamingMode::Buffered,
        metadata: MetadataMap::new(),
    })
}

fn image_analyze_request(request_id: &str, params: ImageAnalyzeParams) -> InferenceRequest {
    InferenceRequest::Generate(GenerateRequest {
        request_id: Some(RequestId::from(request_id)),
        session_id: None,
        run_id: None,
        round_id: None,
        model: ModelSelection {
            specific_model: None,
            required_capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::ImageInput,
            ],
            preferred_capabilities: Vec::new(),
        },
        conversation: Conversation {
            messages: vec![ConversationMessage {
                role: MessageRole::User,
                parts: vec![
                    ContentPart::Image(ImageInput {
                        source: MediaSource::Base64(params.image_data),
                        media_type: None,
                    }),
                    ContentPart::Text(TextPart {
                        text: params.prompt,
                    }),
                ],
                metadata: MetadataMap::new(),
            }],
            metadata: MetadataMap::new(),
        },
        tools: Vec::new(),
        tool_choice: ToolChoice::Disabled,
        reasoning: ReasoningSettings {
            thinking: ThinkingMode::Disabled,
            effort: None,
        },
        output_constraint: OutputConstraint::None,
        sampling: SamplingConfig {
            max_output_tokens: Some(params.max_tokens),
            temperature: Some(params.temperature as f32),
            ..SamplingConfig::default()
        },
        streaming: StreamingMode::Buffered,
        metadata: MetadataMap::new(),
    })
}

async fn run_round_response_from_stream(
    stream: InferenceStream,
) -> Result<RunRoundResponse, String> {
    let output = collect_generation(stream).await?;
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
) -> Result<ImageAnalyzeResponse, String> {
    let output = collect_generation(stream).await?;
    Ok(ImageAnalyzeResponse {
        text: output.content,
        tokens_generated: output.usage.map_or(0, |usage| usage.output_tokens),
        inference_time_ms: output
            .performance
            .map_or(0, |performance| performance.total_duration_ms),
    })
}

#[derive(Default)]
struct GenerationOutput {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<TokenUsage>,
    performance: Option<PerformanceMetrics>,
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<ToolCallId>,
    name: Option<String>,
    arguments: String,
}

async fn collect_generation(mut stream: InferenceStream) -> Result<GenerationOutput, String> {
    let mut output = GenerationOutput::default();
    let mut partial_tool_calls = BTreeMap::<usize, PartialToolCall>::new();

    while let Some(response) = stream.next().await {
        match response.map_err(|error| error.to_string())? {
            InferenceResponse::GenerationStarted(_) => {}
            InferenceResponse::GenerationChunk(chunk) => {
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
                output.usage = completed.usage;
                output.performance = completed.performance;
            }
            InferenceResponse::Failure(failure) => return Err(failure.message),
            other => return Err(format!("Unsupported inference response: {other:?}")),
        }
    }

    if output.tool_calls.is_empty() && !partial_tool_calls.is_empty() {
        output.tool_calls = partial_tool_calls
            .into_iter()
            .filter_map(|(index, partial)| partial.into_call(index))
            .collect();
    }

    Ok(output)
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

    use nexo_ai::{AutoModelLoader, ModelDataType, ModelLoader};
    use nexo_core::{ModelModalities, RoleStrategy, SupportedModality};

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
                reasoning: ReasoningSettings::default(),
                model_id: None,
            },
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
    fn startup_selection_deduplicates_model_capabilities() {
        let models = LoadedModels::new(
            RuntimeConfig::default(),
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

    fn model_config(id: &str, capabilities: Vec<ModelCapability>) -> RegisteredModelConfig {
        RegisteredModelConfig {
            descriptor: ModelDescriptor {
                id: ModelId::from(id),
                display_name: id.to_string(),
                provider: Some("test".to_string()),
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
            loader: ModelLoader::Auto(AutoModelLoader {
                model_id: id.to_string(),
                from_uqff: None,
                tokenizer_json: None,
                chat_template: None,
                jinja_explicit: None,
                dtype: ModelDataType::Auto,
                hf_cache_path: None,
            }),
            revision: None,
        }
    }
}
