use crate::config::NodeConfig;
use crate::kv_cache::manager::SessionCacheManager;
use crate::registry::ToolRegistry;
use base64::Engine;
use cli_helpers::Error;
use nexo_ai::api::types::{
    ChatMessage, ChatRequest, ChatRole, ImageAnalysisRequest, ModelCategory, ToolCallRequest,
};
use nexo_ai::coordinator::Coordinator;
use nexo_ai::registry::find_manifest;
use nexo_ws_client::{
    NexoConnection, ReadHalf, WriteHalf, default_node_connect_params, perform_handshake,
};
use nexo_ws_schema::{
    ErrorPayload, Frame, ImageAnalyzeParams, LoadedModelInfo, Method, ModelLoadResponse,
    ModelStatusParams, ModelUnloadResponse, ToolsExecuteParams, ToolsExecuteResponse,
    ToolsRegisterParams,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_PREFILL_CACHE_ENTRIES: usize = 64;

/// Run the node, connecting to the gateway and reconnecting on disconnect.
pub async fn run_node(
    config: &NodeConfig,
    available_models: &[String],
    registry: &ToolRegistry,
    coordinator: Arc<Mutex<Coordinator>>,
) -> cli_helpers::Result {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        tracing::info!(
            "Connecting to gateway at {} (attempt {attempt})",
            config.gateway_url
        );

        match connect_and_run(config, available_models, registry, coordinator.clone()).await {
            Ok(()) => {
                tracing::info!("Node disconnected gracefully");
                break;
            }
            Err(e) => {
                tracing::warn!(
                    "Connection lost: {e}. Reconnecting in {}ms...",
                    config.reconnect_interval_ms
                );
                tokio::time::sleep(Duration::from_millis(config.reconnect_interval_ms)).await;
            }
        }
    }
    Ok(())
}

async fn connect_and_run(
    config: &NodeConfig,
    available_models: &[String],
    registry: &ToolRegistry,
    coordinator: Arc<Mutex<Coordinator>>,
) -> cli_helpers::Result {
    // Step 1: Connect to gateway
    let mut conn = NexoConnection::connect(&config.gateway_url, &config.auth_token)
        .await
        .map_err(|e| Error::Network(format!("Connection failed: {e}")))?;

    tracing::info!("Connected to gateway");

    // Step 2: Handshake — declare capabilities and available models
    let (capabilities, commands) = registry.capabilities_and_commands();

    tracing::debug!("Handshaking with capabilities={capabilities:?}, commands={commands:?}");

    let params = default_node_connect_params(
        &config.node_id,
        &config.node_version,
        config.platform,
        &config.device_id,
        capabilities,
        commands,
        available_models.to_vec(),
    );

    let hello = perform_handshake(&mut conn, params)
        .await
        .map_err(|e| Error::Network(format!("Handshake failed: {e}")))?;

    tracing::info!(
        "Handshake complete: protocol v{}, tick interval {}ms",
        hello.protocol,
        hello.policy.tick_interval_ms
    );

    // Step 3: Register tools with full specs
    let specs = registry.specs();
    let tool_count = specs.len();
    tracing::info!("Registering {tool_count} tool(s) with gateway");

    let register_frame =
        Frame::request(Method::ToolsRegister, &ToolsRegisterParams { tools: specs })
            .map_err(|e| Error::Other(format!("Failed to build register frame: {e}")))?;

    conn.send_frame(&register_frame)
        .await
        .map_err(|e| Error::Network(format!("Failed to send register: {e}")))?;

    // Wait for register response, skipping any events that arrive first
    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|e| Error::Network(format!("Failed to receive register response: {e}")))?
            .ok_or_else(|| Error::Network("Connection closed during registration".into()))?;

        match frame {
            Frame::Response {
                ok: true, payload, ..
            } => {
                let registered = payload
                    .as_ref()
                    .and_then(|p| p.get("registered"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                tracing::info!("Gateway accepted {registered}/{tool_count} tool(s)");
                break;
            }
            Frame::Response {
                ok: false, error, ..
            } => {
                let msg = error
                    .map(|e| format!("{}: {}", e.code, e.message))
                    .unwrap_or_else(|| "Unknown error".into());
                return Err(Error::Network(format!("Tool registration rejected: {msg}")));
            }
            Frame::Event { .. } => continue,
            other => {
                tracing::warn!("Unexpected frame during registration: {other:?}");
            }
        }
    }

    tracing::info!("Node ready, listening for requests");

    // Split connection for non-blocking inference
    let (mut writer, mut reader) = conn.into_split();

    // Push initial model status so the gateway knows what's loaded.
    push_model_status(&mut writer, &coordinator, available_models).await;
    let mut prefill_cache = std::collections::HashMap::<String, String>::new();
    let cache_manager = Arc::new(Mutex::new(SessionCacheManager::new(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".nexo")
            .join("kv_cache"),
    )));

    // Inference result channel: (request_id, result_json_or_error)
    let (inference_tx, mut inference_rx) =
        tokio::sync::mpsc::channel::<(String, Result<serde_json::Value, String>)>(1);
    let mut inference_busy = false;

    // Step 4: Message loop with select!
    loop {
        tokio::select! {
            frame = reader.recv_frame() => {
                let frame = frame
                    .map_err(|e| Error::Network(format!("Receive error: {e}")))?;
                match frame {
                    Some(Frame::Request {
                        id,
                        method: Method::ToolsExecute,
                        params,
                    }) => {
                        handle_tool_execute(&mut writer, &id, params, registry).await?;
                    }
                    Some(Frame::Request {
                        id,
                        method: Method::Agent,
                        params,
                    }) => {
                        if inference_busy {
                            send_busy_error(&mut writer, &id).await?;
                        } else {
                            inference_busy = true;
                            dispatch_agent_inference(
                                &id,
                                params,
                                &coordinator,
                                &inference_tx,
                                &mut prefill_cache,
                                &mut reader,
                                &mut writer,
                                &cache_manager,
                            ).await?;
                        }
                    }
                    Some(Frame::Request {
                        id,
                        method: Method::ImageAnalyze,
                        params,
                    }) => {
                        if inference_busy {
                            send_busy_error(&mut writer, &id).await?;
                        } else {
                            inference_busy = true;
                            dispatch_image_analyze(
                                &id,
                                params,
                                &coordinator,
                                &inference_tx,
                            ).await?;
                        }
                    }
                    Some(Frame::Request {
                        id,
                        method: Method::ModelLoad,
                        params,
                    }) => {
                        handle_model_load(&mut writer, &id, params, &coordinator, available_models).await?;
                    }
                    Some(Frame::Request {
                        id,
                        method: Method::ModelUnload,
                        params,
                    }) => {
                        handle_model_unload(&mut writer, &id, params, &coordinator, available_models, &cache_manager).await?;
                    }
                    Some(Frame::Event {
                        event: nexo_ws_schema::EventKind::Tick,
                        ..
                    }) => {
                        tracing::trace!("Received tick");
                    }
                    Some(Frame::Event {
                        event: nexo_ws_schema::EventKind::Shutdown,
                        payload,
                        ..
                    }) => {
                        let reason = payload
                            .get("reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        tracing::info!("Received shutdown event: {reason}");
                        break;
                    }
                    Some(Frame::Event { event, .. }) => {
                        tracing::debug!("Received event: {event:?}");
                    }
                    Some(frame) => {
                        tracing::debug!("Received unexpected frame: {frame:?}");
                    }
                    None => {
                        return Err(Error::Network(
                            "Connection closed by gateway".into(),
                        ));
                    }
                }
            }

            // Inference completion branch
            Some((request_id, result)) = inference_rx.recv() => {
                inference_busy = false;

                // Periodically expire old KV caches from disk
                {
                    let mut mgr = cache_manager.lock().unwrap();
                    if let Err(e) = mgr.maybe_expire() {
                        tracing::warn!("KV cache expiry failed: {e}");
                    }
                }

                let response = match result {
                    Ok(payload) => {
                        Frame::ok_response(&request_id, &payload).unwrap_or_else(|e| {
                            Frame::error_response(
                                &request_id,
                                ErrorPayload {
                                    code: "internal_error".into(),
                                    message: e.to_string(),
                                },
                            )
                        })
                    }
                    Err(err_msg) => {
                        Frame::error_response(
                            &request_id,
                            ErrorPayload {
                                code: "inference_error".into(),
                                message: err_msg,
                            },
                        )
                    }
                };
                send(&mut writer, &response).await?;
            }
        }
    }

    if let Err(e) = writer.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }

    Ok(())
}

// ── Tool execution (synchronous, fast) ────────────────────────────────────

async fn handle_tool_execute(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    registry: &ToolRegistry,
) -> cli_helpers::Result {
    let exec_params: ToolsExecuteParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => {
            let error_response = Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "invalid_params".into(),
                    message: format!("Invalid tools.execute params: {e}"),
                },
            );
            send(writer, &error_response).await?;
            return Ok(());
        }
    };

    tracing::info!("Executing tool '{}'", exec_params.tool);
    tracing::debug!("Tool '{}' args: {}", exec_params.tool, exec_params.args);
    let start = std::time::Instant::now();

    let response = match registry.execute(&exec_params.tool, exec_params.args).await {
        Some(result) => {
            let elapsed = start.elapsed();
            tracing::info!(
                "Tool '{}' completed in {:.2}ms (success={})",
                exec_params.tool,
                elapsed.as_secs_f64() * 1000.0,
                result.success
            );
            tracing::debug!(
                "Tool '{}' output: {}, error: {:?}",
                exec_params.tool,
                result.output,
                result.error
            );
            Frame::ok_response(
                request_id,
                &ToolsExecuteResponse {
                    success: result.success,
                    output: result.output,
                    error: result.error,
                },
            )
            .unwrap_or_else(|e| {
                Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "internal_error".into(),
                        message: e.to_string(),
                    },
                )
            })
        }
        None => {
            tracing::warn!("Tool '{}' not found locally", exec_params.tool);
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "tool_not_found".into(),
                    message: format!("Tool '{}' is not available on this node", exec_params.tool),
                },
            )
        }
    };

    send(writer, &response).await?;

    Ok(())
}

// ── Agent inference dispatch (async → spawn_blocking) ─────────────────────
#[allow(clippy::too_many_arguments)]
async fn dispatch_agent_inference(
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    tx: &tokio::sync::mpsc::Sender<(String, Result<serde_json::Value, String>)>,
    prefill_cache: &mut HashMap<String, String>,
    reader: &mut ReadHalf,
    writer: &mut WriteHalf,
    cache_manager: &Arc<Mutex<SessionCacheManager>>,
) -> cli_helpers::Result {
    let mut messages: Vec<serde_json::Value> = params
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let tools: Vec<serde_json::Value> = params
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    // Accept both OpenAI format {"type":"function","function":{...}}
                    // and flat format {"name":...,"description":...,"parameters":...}
                    t.get("function").cloned().or_else(|| Some(t.clone()))
                })
                .collect()
        })
        .unwrap_or_default();

    let prefill_sha = params
        .get("prefill_sha")
        .and_then(|v| v.as_str())
        .map(String::from);

    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let session_id = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Resolve prefill by SHA, prepending as system message if present
    if let Some(sha) = &prefill_sha {
        let content: String = if let Some(cached) = prefill_cache.get(sha.as_str()) {
            cached.clone()
        } else {
            match fetch_prefill_from_gateway(reader, writer, sha).await {
                Ok(content) => {
                    if prefill_cache.len() >= MAX_PREFILL_CACHE_ENTRIES {
                        // Evict an arbitrary entry to stay bounded.
                        if let Some(key) = prefill_cache.keys().next().cloned() {
                            prefill_cache.remove(&key);
                        }
                    }
                    prefill_cache.insert(sha.clone(), content.clone());
                    content
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch prefill '{sha}': {e}");
                    String::new()
                }
            }
        };
        if !content.is_empty() {
            messages.insert(
                0,
                serde_json::json!({ "role": "system", "content": content }),
            );
        }
    }

    // Convert to typed ChatMessages
    let chat_messages: Vec<ChatMessage> = messages
        .iter()
        .map(|m| {
            let role = match m.get("role").and_then(|v| v.as_str()) {
                Some("system") => ChatRole::System,
                Some("assistant") => ChatRole::Assistant,
                Some("tool") => ChatRole::Tool,
                _ => ChatRole::User,
            };
            let content = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            ChatMessage { role, content }
        })
        .collect();

    // Log the last message for conversation tracking
    if let Some(last) = chat_messages.last() {
        tracing::info!(
            "Inference request {}: last message role={:?}, content='{:.120}'",
            request_id,
            last.role,
            last.content,
        );
    }

    // Determine which model to use
    tracing::debug!(
        "dispatch_agent_inference {}: resolving model (model_id={:?}, has_tools={}, session_id={:?}, messages={})",
        request_id,
        model_id,
        !tools.is_empty(),
        session_id,
        chat_messages.len(),
    );
    let coord = coordinator.lock()?;
    let model_name = if let Some(id) = &model_id {
        id.clone()
    } else if !tools.is_empty() {
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
    drop(coord);

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

    // Spawn blocking inference
    let coord = coordinator.clone();
    let cache_mgr = cache_manager.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();
    let has_tools = !tools.is_empty();

    // Gemma 4 recommended defaults
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
            let mut coord = coord.lock().unwrap();
            tracing::debug!(
                "dispatch_agent_inference {}: coordinator lock acquired",
                request_id
            );
            let model = coord
                .model_mut(&model_name)
                .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

            // Handle KV cache session switching (disk persistence)
            if let Some(kv) = model.as_kv_cacheable() {
                let target = session_id.as_deref();
                tracing::debug!(
                    "dispatch_agent_inference {}: switching KV cache to target={:?}",
                    request_id,
                    target,
                );

                let mgr = cache_mgr.lock().unwrap();
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

            // Re-borrow model after kv_cacheable borrow ends
            let model = coord
                .model_mut(&model_name)
                .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

            tracing::debug!(
                "dispatch_agent_inference {}: dispatching to {} path",
                request_id,
                if has_tools { "tool-call" } else { "chat" },
            );

            if has_tools {
                let tool_specs: Vec<nexo_spec::tool::ToolSpec> =
                    serde_json::from_value(serde_json::Value::Array(tools))
                        .map_err(|e| format!("Failed to parse tool specs: {e}"))?;

                let req = ToolCallRequest {
                    messages: chat_messages.clone(),
                    tools: tool_specs,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    session_id: session_id.clone(),
                };

                if let Some(kv) = model.as_kv_cacheable() {
                    kv.clear_kv_cache();
                    kv.set_session_state(session_id.clone(), Vec::new());
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
                let resp = tool_model.call_tools(&req).map_err(|e| e.to_string())?;

                tracing::debug!(
                    "dispatch_agent_inference {}: tool inference complete, {} tool calls",
                    request_id,
                    resp.tool_calls.len()
                );

                if let Some(kv) = model.as_kv_cacheable() {
                    kv.clear_kv_cache();
                    kv.set_session_state(session_id.clone(), Vec::new());
                    tracing::debug!(
                        "dispatch_agent_inference {}: reset in-memory KV state after tool selection inference",
                        request_id,
                    );
                }

                if resp.tool_calls.is_empty() {
                    let fallback_text = resp.reasoning.unwrap_or_default();
                    tracing::debug!(
                        "dispatch_agent_inference {}: no tool calls returned, falling back to chat response",
                        request_id,
                    );

                    let chat_req = ChatRequest {
                        messages: chat_messages.clone(),
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        session_id: session_id.clone(),
                    };
                    let Some(chat_model) = model.as_chat() else {
                        return Ok(serde_json::json!({ "content": fallback_text }));
                    };
                    let chat_resp = chat_model.chat(&chat_req).map_err(|e| e.to_string())?;

                    if let Some(kv) = model.as_kv_cacheable() {
                        kv.clear_kv_cache();
                        kv.set_session_state(session_id.clone(), Vec::new());
                        tracing::debug!(
                            "dispatch_agent_inference {}: reset in-memory KV state after chat fallback response",
                            request_id,
                        );
                    }

                    Ok(serde_json::json!({ "content": chat_resp.text }))
                } else {
                    let calls: Vec<serde_json::Value> = resp
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": nexo_ws_schema::Frame::new_id(),
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                            })
                        })
                        .collect();
                    Ok(serde_json::json!({ "tool_calls": calls }))
                }
            } else {
                let req = ChatRequest {
                    messages: chat_messages,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    session_id,
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
                let resp = chat_model.chat(&req).map_err(|e| e.to_string())?;

                tracing::debug!(
                    "dispatch_agent_inference {}: chat inference complete, response length={}",
                    request_id,
                    resp.text.len()
                );

                Ok(serde_json::json!({ "content": resp.text }))
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

// ── Image analysis dispatch (async → spawn_blocking) ──────────────────────

async fn dispatch_image_analyze(
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    tx: &tokio::sync::mpsc::Sender<(String, Result<serde_json::Value, String>)>,
) -> cli_helpers::Result {
    let analyze_params: ImageAnalyzeParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => {
            let _ = tx
                .send((
                    request_id.to_string(),
                    Err(format!("Invalid image.analyze params: {e}")),
                ))
                .await;
            return Ok(());
        }
    };

    tracing::info!("Analyzing image (prompt: '{:.80}')", analyze_params.prompt);

    // Determine model
    let coord = coordinator.lock()?;
    let model_name = coord
        .active_model_for(ModelCategory::Image)
        .unwrap_or_default()
        .to_string();
    drop(coord);

    let coord = coordinator.clone();
    let tx = tx.clone();
    let request_id = request_id.to_string();

    // Base64 decode + inference both run on the blocking pool
    tokio::task::spawn_blocking(move || {
        let result = (|| -> Result<serde_json::Value, String> {
            let image_bytes = base64::engine::general_purpose::STANDARD
                .decode(&analyze_params.image_data)
                .map_err(|e| format!("Invalid base64 image data: {e}"))?;

            let mut coord = coord.lock().unwrap();
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

            let resp = image_model.analyze_image(&req).map_err(|e| e.to_string())?;

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

// ── Model load/unload (blocking but fast — weight loading) ────────────────

async fn handle_model_load(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
) -> cli_helpers::Result {
    let model_id = params
        .get("modelId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    tracing::info!("Loading model '{model_id}'");

    let coord = coordinator.clone();
    let model_id_clone = model_id.clone();
    let (loaded, error) = tokio::task::spawn_blocking(move || {
        let mut coord = coord.lock().unwrap();
        match coord.load_model(&model_id_clone) {
            Ok(()) => (true, None),
            Err(e) => {
                tracing::error!("Failed to load model '{model_id_clone}': {e}");
                (false, Some(e.to_string()))
            }
        }
    })
    .await
    .unwrap_or((false, Some("Task panicked".into())));

    let response = Frame::ok_response(
        request_id,
        &ModelLoadResponse {
            model_id: model_id.clone(),
            loaded,
            error,
        },
    )
    .unwrap_or_else(|e| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: e.to_string(),
            },
        )
    });

    send(writer, &response).await?;

    if loaded {
        // Set this model as active for all its supported categories.
        if let Some(manifest) = find_manifest(&model_id) {
            let mut coord = coordinator.lock()?;
            for cat in &manifest.categories {
                coord.set_active_model(*cat, model_id.clone());
            }
        }
        push_model_status(writer, coordinator, available_models).await;
    }

    Ok(())
}

async fn handle_model_unload(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
    cache_manager: &Arc<Mutex<SessionCacheManager>>,
) -> cli_helpers::Result {
    let model_id = params
        .get("modelId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    tracing::info!("Unloading model '{model_id}'");

    let coord = coordinator.clone();
    let cache_mgr = cache_manager.clone();
    let model_id_clone = model_id.clone();
    let unloaded = tokio::task::spawn_blocking(move || {
        let mut coord = coord.lock().unwrap();

        // Save current session's KV cache to disk before unloading
        if let Some(model) = coord.model_mut(&model_id_clone)
            && let Some(kv) = model.as_kv_cacheable()
        {
            let mgr = cache_mgr.lock().unwrap();
            if let Err(e) = mgr.on_model_unload(&model_id_clone, kv) {
                tracing::warn!("Failed to save KV cache before unload: {e}");
            }
        }

        match coord.unload_model(&model_id_clone) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("Unload of model '{model_id_clone}' failed (non-fatal): {e}");
                false
            }
        }
    })
    .await
    .unwrap_or(false);

    let response = Frame::ok_response(request_id, &ModelUnloadResponse { unloaded })
        .unwrap_or_else(|e| {
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "internal_error".into(),
                    message: e.to_string(),
                },
            )
        });

    send(writer, &response).await?;

    if unloaded {
        // Clear active model for categories that were served by this model.
        if let Some(manifest) = find_manifest(&model_id) {
            let mut coord = coordinator.lock()?;
            for cat in &manifest.categories {
                if coord.active_model_for(*cat).is_some_and(|m| m == model_id) {
                    coord.remove_active_model(*cat);
                }
            }
        }
    }

    push_model_status(writer, coordinator, available_models).await;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Send a frame, mapping the WS error to `Error::Network`.
async fn send(writer: &mut WriteHalf, frame: &Frame) -> cli_helpers::Result {
    writer
        .send_frame(frame)
        .await
        .map_err(|e| Error::Network(format!("Send error: {e}")))
}

/// Send a "node_busy" error response when inference is already in progress.
async fn send_busy_error(writer: &mut WriteHalf, request_id: &str) -> cli_helpers::Result {
    let err = Frame::error_response(
        request_id,
        ErrorPayload {
            code: "node_busy".into(),
            message: "Inference is already in progress".into(),
        },
    );
    send(writer, &err).await
}

/// Push the node's current loaded model state to the gateway.
async fn push_model_status(
    writer: &mut WriteHalf,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
) {
    let loaded_models: Vec<LoadedModelInfo> = {
        let coord = coordinator.lock().unwrap();
        coord
            .loaded_models()
            .iter()
            .map(|(name, cats)| LoadedModelInfo {
                model_id: name.to_string(),
                categories: cats.to_vec(),
            })
            .collect()
    };
    let status = ModelStatusParams {
        loaded_models,
        available_models: available_models.to_vec(),
    };
    if let Ok(frame) = Frame::request(Method::ModelStatus, &status) {
        let _ = writer.send_frame(&frame).await;
    }
}

/// Fetch prefill content from the gateway by SHA.
/// Sends a PrefillFetch request and waits for the response (up to 10s).
async fn fetch_prefill_from_gateway(
    reader: &mut ReadHalf,
    writer: &mut WriteHalf,
    prefill_sha: &str,
) -> anyhow::Result<String> {
    let fetch_id = Frame::new_id();
    let fetch_frame = Frame::Request {
        id: fetch_id.clone(),
        method: Method::PrefillFetch,
        params: serde_json::json!({ "prefillSha": prefill_sha }),
    };
    writer
        .send_frame(&fetch_frame)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to send PrefillFetch: {e}"))?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            anyhow::bail!("PrefillFetch timed out for '{prefill_sha}'");
        }

        let frame = tokio::time::timeout(remaining, reader.recv_frame())
            .await
            .map_err(|_| anyhow::anyhow!("PrefillFetch timed out for '{prefill_sha}'"))?
            .map_err(|e| anyhow::anyhow!("Connection error during PrefillFetch: {e}"))?
            .ok_or_else(|| anyhow::anyhow!("Connection closed during PrefillFetch"))?;

        match frame {
            Frame::Response {
                id,
                ok: true,
                payload,
                ..
            } if id == fetch_id => {
                let content = payload
                    .as_ref()
                    .and_then(|p| p.get("content"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Missing 'content' field in PrefillFetch response")
                    })?;
                return Ok(content);
            }
            Frame::Response {
                id,
                ok: false,
                error,
                ..
            } if id == fetch_id => {
                let msg = error
                    .map(|e| format!("{}: {}", e.code, e.message))
                    .unwrap_or_else(|| "Unknown error".into());
                anyhow::bail!("PrefillFetch failed for sha '{prefill_sha}': {msg}");
            }
            other => {
                tracing::debug!(
                    "Discarding non-prefill frame while awaiting PrefillFetch: {other:?}"
                );
            }
        }
    }
}
