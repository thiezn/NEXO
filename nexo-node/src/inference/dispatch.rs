use crate::inference::SessionCacheManager;
use base64::Engine;
use cli_helpers::Error;
use nexo_ai::api::types::{
    ChatMessage, ChatRequest, ChatRole, ImageAnalysisRequest, ModelCategory, ToolCallRequest,
};
use nexo_ai::coordinator::Coordinator;
use nexo_ws_schema::{
    AgentRoundRequest, AgentRoundResponse, AgentRoundToolCall, Frame, ImageAnalyzeParams,
};
use std::sync::{Arc, Mutex, MutexGuard};

type InferenceSender = tokio::sync::mpsc::Sender<(String, Result<serde_json::Value, String>)>;

fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Result<MutexGuard<'a, T>, String> {
    mutex
        .lock()
        .map_err(|error| format!("Failed to lock {name}: {error}"))
}

/// Queue an agent inference round on the blocking pool and send the result back by channel.
pub(crate) async fn dispatch_agent_inference(
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    tx: &InferenceSender,
    cache_manager: &Arc<Mutex<SessionCacheManager>>,
) -> cli_helpers::Result {
    let round_request: AgentRoundRequest = match serde_json::from_value(params) {
        Ok(request) => request,
        Err(error) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid typed agent round params: {error}")),
                ))
                .await;
            return Ok(());
        }
    };

    let chat_messages: Vec<ChatMessage> = round_request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role.as_str() {
                "system" => ChatRole::System,
                "assistant" => ChatRole::Assistant,
                "tool" => ChatRole::Tool,
                _ => ChatRole::User,
            };
            ChatMessage::with_tool_metadata(
                role,
                message.content.clone(),
                message.tool_call_id.clone(),
                message.tool_name.clone(),
            )
        })
        .collect();

    if let Some(last) = chat_messages.last() {
        tracing::info!(
            "Inference request {}: last message role={:?}, content='{:.120}'",
            request_id,
            last.role,
            last.content,
        );
    }

    tracing::debug!(
        "dispatch_agent_inference {}: resolving model (model_id={:?}, has_tools={}, session_id={:?}, messages={})",
        request_id,
        round_request.model_id,
        !round_request.tools.is_empty(),
        round_request.session_id,
        chat_messages.len(),
    );
    let (model_name, settings) = {
        let coord = lock_mutex(coordinator.as_ref(), "coordinator").map_err(Error::Other)?;
        let model_name = if let Some(id) = &round_request.model_id {
            id.clone()
        } else if !round_request.tools.is_empty() {
            coord
                .active_model_for(ModelCategory::Tool)
                .or_else(|| coord.active_model_for(ModelCategory::Chat))
                .unwrap_or_default()
                .to_string()
        } else {
            coord
                .active_model_for(ModelCategory::Chat)
                .unwrap_or_default()
                .to_string()
        };
        let settings = coord.config().model_settings(&model_name);
        (model_name, settings)
    };

    let session_id = round_request.session_id.clone();
    let run_id = round_request.run_id.clone();
    let round_id = round_request.round_id.clone();
    let tool_specs = round_request.tools.clone();

    if model_name.is_empty() {
        tracing::warn!(
            "dispatch_agent_inference {}: no model available",
            request_id
        );
        let _ = tx
            .send((
                request_id.to_string(),
                Err("No model loaded for inference".into()),
            ))
            .await;
        return Ok(());
    }

    tracing::debug!(
        "dispatch_agent_inference {}: using model '{}', temperature={}, top_p={}, top_k={:?}, max_tokens={}",
        request_id,
        model_name,
        settings.temperature.unwrap_or(1.0),
        settings.top_p.unwrap_or(0.95),
        settings.top_k.or(Some(64)),
        settings.max_tokens.unwrap_or(2048),
    );

    let coord = coordinator.clone();
    let cache_mgr = cache_manager.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();
    let run_id_for_task = run_id.clone();
    let round_id_for_task = round_id.clone();
    let has_tools = !tool_specs.is_empty();

    let temperature = settings.temperature.unwrap_or(1.0);
    let top_p = settings.top_p.unwrap_or(0.95);
    let top_k = settings.top_k.or(Some(64));
    let max_tokens = settings.max_tokens.unwrap_or(2048);

    tokio::task::spawn_blocking(move || {
        tracing::debug!(
            "dispatch_agent_inference {}: spawn_blocking started",
            request_id
        );
        let result = (|| -> Result<serde_json::Value, String> {
            tracing::debug!(
                "dispatch_agent_inference {}: acquiring coordinator lock",
                request_id
            );
            let mut coord = lock_mutex(coord.as_ref(), "coordinator")?;
            tracing::debug!(
                "dispatch_agent_inference {}: coordinator lock acquired",
                request_id
            );
            let model = coord
                .model_mut(&model_name)
                .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

            if let Some(kv) = model.as_kv_cacheable() {
                let target = Some(session_id.as_str());
                tracing::debug!(
                    "dispatch_agent_inference {}: switching KV cache to target={:?}",
                    request_id,
                    target,
                );

                let mgr = lock_mutex(cache_mgr.as_ref(), "session cache manager")?;
                if let Err(error) = mgr.switch_session(&model_name, kv, target) {
                    tracing::warn!(
                        "Failed to switch KV cache for model '{}' to session {:?}: {error}",
                        model_name,
                        target,
                    );
                }
            }

            tracing::debug!(
                "dispatch_agent_inference {}: KV cache handling complete",
                request_id
            );

            let model = coord
                .model_mut(&model_name)
                .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

            tracing::debug!(
                "dispatch_agent_inference {}: dispatching to {} path",
                request_id,
                if has_tools { "tool-call" } else { "chat" },
            );

            if has_tools {
                let req = ToolCallRequest {
                    messages: chat_messages.clone(),
                    tools: tool_specs.clone(),
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    session_id: Some(session_id.clone()),
                };

                if let Some(kv) = model.as_kv_cacheable() {
                    kv.clear_kv_cache();
                    kv.set_session_state(Some(session_id.clone()), Vec::new());
                    tracing::debug!(
                        "dispatch_agent_inference {}: cleared in-memory KV state before tool selection inference",
                        request_id,
                    );
                }

                let tool_model = model
                    .as_tool()
                    .ok_or_else(|| format!("Model '{model_name}' does not support tool calling"))?;
                tracing::debug!(
                    "dispatch_agent_inference {}: starting tool inference ({} tools, {} messages)",
                    request_id,
                    req.tools.len(),
                    req.messages.len()
                );
                let resp = tool_model
                    .call_tools(&req)
                    .map_err(|error| error.to_string())?;

                tracing::debug!(
                    "dispatch_agent_inference {}: tool inference complete, {} tool calls",
                    request_id,
                    resp.tool_calls.len()
                );

                if let Some(kv) = model.as_kv_cacheable() {
                    kv.clear_kv_cache();
                    kv.set_session_state(Some(session_id.clone()), Vec::new());
                    tracing::debug!(
                        "dispatch_agent_inference {}: reset in-memory KV state after tool selection inference",
                        request_id,
                    );
                }

                let response = AgentRoundResponse {
                    content: if resp.tool_calls.is_empty() {
                        resp.reasoning.clone().filter(|text| !text.is_empty())
                    } else {
                        None
                    },
                    rationale: resp.reasoning,
                    tool_calls: resp
                        .tool_calls
                        .into_iter()
                        .map(|tool_call| AgentRoundToolCall {
                            id: Frame::new_id(),
                            name: tool_call.name,
                            arguments: tool_call.arguments,
                        })
                        .collect(),
                };
                tracing::info!(
                    "Agent round {round_id_for_task} for run {run_id_for_task} completed on tool path: tool_calls={}, content_chars={}, reasoning_chars={}",
                    response.tool_calls.len(),
                    response.content.as_deref().map_or(0, str::len),
                    response.rationale.as_deref().map_or(0, str::len),
                );
                Ok(serde_json::to_value(response).map_err(|error| error.to_string())?)
            } else {
                let req = ChatRequest {
                    messages: chat_messages,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    session_id: Some(session_id.clone()),
                };

                let chat_model = model
                    .as_chat()
                    .ok_or_else(|| format!("Model '{model_name}' does not support chat"))?;
                tracing::debug!(
                    "dispatch_agent_inference {}: starting chat inference ({} messages, max_tokens={})",
                    request_id,
                    req.messages.len(),
                    req.max_tokens
                );
                let resp = chat_model.chat(&req).map_err(|error| error.to_string())?;

                tracing::debug!(
                    "dispatch_agent_inference {}: chat inference complete, response length={}",
                    request_id,
                    resp.text.len()
                );

                Ok(serde_json::to_value(AgentRoundResponse {
                    content: Some(resp.text),
                    rationale: None,
                    tool_calls: Vec::new(),
                })
                .map_err(|error| error.to_string())?)
            }
        })();
        tracing::debug!(
            "dispatch_agent_inference {}: sending result (ok={})",
            request_id,
            result.is_ok(),
        );
        let _ = tx.blocking_send((request_id, result));
    });

    Ok(())
}

/// Queue an image-analysis request on the blocking pool and return the result by channel.
pub(crate) async fn dispatch_image_analyze(
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    tx: &InferenceSender,
) -> cli_helpers::Result {
    let analyze_params: ImageAnalyzeParams = match serde_json::from_value(params) {
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

    tracing::info!("Analyzing image (prompt: '{:.80}')", analyze_params.prompt);

    let coord = lock_mutex(coordinator.as_ref(), "coordinator").map_err(Error::Other)?;
    let model_name = coord
        .active_model_for(ModelCategory::Image)
        .unwrap_or_default()
        .to_string();
    drop(coord);

    let coord = coordinator.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    tokio::task::spawn_blocking(move || {
        let result = (|| -> Result<serde_json::Value, String> {
            let image_bytes = base64::engine::general_purpose::STANDARD
                .decode(&analyze_params.image_data)
                .map_err(|error| format!("Invalid base64 image data: {error}"))?;

            let mut coord = lock_mutex(coord.as_ref(), "coordinator")?;
            let model = coord
                .model_mut(&model_name)
                .ok_or_else(|| format!("Image model '{model_name}' not loaded"))?;

            let image_model = model
                .as_image()
                .ok_or_else(|| format!("Model '{model_name}' does not support image analysis"))?;

            let req = ImageAnalysisRequest {
                image_data: image_bytes,
                prompt: analyze_params.prompt,
                max_tokens: analyze_params.max_tokens,
                temperature: analyze_params.temperature,
            };

            let resp = image_model
                .analyze_image(&req)
                .map_err(|error| error.to_string())?;

            tracing::debug!("Raw image analysis response: {}", resp.text);

            Ok(serde_json::json!({
                "text": resp.text,
                "tokensGenerated": resp.tokens_generated,
                "inferenceTimeMs": resp.inference_time_ms,
            }))
        })();
        let _ = tx.blocking_send((request_id, result));
    });

    Ok(())
}
