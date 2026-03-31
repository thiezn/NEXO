use crate::config::NodeConfig;
use crate::download::registry::DEFAULT_INFERENCE_MODEL;
use crate::inference_clients::{
    ChatMessage, ChatRequest, ChatRole, InferenceClients, InferenceConfig, ToolCallRequest,
};
use crate::registry::ToolRegistry;
use nexo_ws_client::{NexoConnection, default_node_connect_params, perform_handshake};
use nexo_ws_schema::{
    ErrorPayload, Frame, Method, ModelLoadResponse, ModelStatusParams, ModelUnloadResponse,
    ToolsExecuteParams, ToolsExecuteResponse, ToolsRegisterParams,
};
use std::collections::HashMap;
use std::time::Duration;

/// Run the node, connecting to the gateway and reconnecting on disconnect.
pub async fn run_node(
    config: &NodeConfig,
    registry: &ToolRegistry,
    has_inference: bool,
) -> utl_helpers::Result {
    let inference = InferenceClients::new(InferenceConfig::default());
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        tracing::info!(
            "Connecting to gateway at {} (attempt {attempt})",
            config.gateway_url
        );

        match connect_and_run(config, registry, &inference, has_inference).await {
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
    registry: &ToolRegistry,
    inference: &InferenceClients,
    has_inference: bool,
) -> utl_helpers::Result {
    // Step 1: Connect to gateway
    let mut conn = NexoConnection::connect(&config.gateway_url, &config.auth_token)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Connection failed: {e}")))?;

    tracing::info!("Connected to gateway");

    // Step 2: Handshake — declare capabilities and available models
    let (mut capabilities, commands) = registry.capabilities_and_commands();
    if has_inference {
        capabilities.push("llm".to_string());
    }
    tracing::debug!(
        "Handshaking with capabilities={capabilities:?}, commands={commands:?}"
    );

    let params = default_node_connect_params(
        &config.node_id,
        &config.node_version,
        config.platform,
        &config.device_id,
        capabilities,
        commands,
        config.available_models.clone(),
    );

    let hello = perform_handshake(&mut conn, params)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Handshake failed: {e}")))?;

    tracing::info!(
        "Handshake complete: protocol v{}, tick interval {}ms",
        hello.protocol,
        hello.policy.tick_interval_ms
    );

    // Step 3: Register tools with full specs
    let specs = registry.specs();
    let tool_count = specs.len();
    tracing::info!("Registering {tool_count} tool(s) with gateway");

    let register_frame = Frame::request(
        Method::ToolsRegister,
        &ToolsRegisterParams { tools: specs },
    )
    .map_err(|e| utl_helpers::Error::Other(format!("Failed to build register frame: {e}")))?;

    conn.send_frame(&register_frame)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Failed to send register: {e}")))?;

    // Wait for register response, skipping any events that arrive first
    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|e| utl_helpers::Error::Network(format!("Failed to receive register response: {e}")))?
            .ok_or_else(|| utl_helpers::Error::Network("Connection closed during registration".into()))?;

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
                return Err(utl_helpers::Error::Network(format!(
                    "Tool registration rejected: {msg}"
                )));
            }
            Frame::Event { .. } => continue,
            other => {
                tracing::warn!("Unexpected frame during registration: {other:?}");
            }
        }
    }

    tracing::info!("Node ready, listening for requests");

    // Push initial model status so the gateway knows inference is available.
    if has_inference {
        let loaded = Some(DEFAULT_INFERENCE_MODEL.to_string());
        push_model_status(&mut conn, loaded, &config.available_models).await;
    }

    // Per-connection state
    let available_models = config.available_models.clone();
    let mut prefill_cache: HashMap<String, String> = HashMap::new();
    let mut currently_loaded: Option<String> = None;

    // Step 4: Message loop
    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|e| utl_helpers::Error::Network(format!("Receive error: {e}")))?;

        match frame {
            Some(Frame::Request {
                id,
                method: Method::ToolsExecute,
                params,
            }) => {
                handle_tool_execute(&mut conn, &id, params, registry).await?;
            }
            Some(Frame::Request {
                id,
                method: Method::Agent,
                params,
            }) => {
                handle_agent_inference(&mut conn, &id, params, inference, &mut prefill_cache)
                    .await?;
            }
            Some(Frame::Request {
                id,
                method: Method::ModelLoad,
                params,
            }) => {
                handle_model_load(&mut conn, &id, params, inference, &mut currently_loaded, &available_models).await?;
            }
            Some(Frame::Request {
                id,
                method: Method::ModelUnload,
                params,
            }) => {
                handle_model_unload(&mut conn, &id, params, inference, &mut currently_loaded, &available_models)
                    .await?;
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
                return Err(utl_helpers::Error::Network(
                    "Connection closed by gateway".into(),
                ));
            }
        }
    }

    if let Err(e) = conn.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }

    Ok(())
}

async fn handle_tool_execute(
    conn: &mut NexoConnection,
    request_id: &str,
    params: serde_json::Value,
    registry: &ToolRegistry,
) -> utl_helpers::Result {
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
            conn.send_frame(&error_response)
                .await
                .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;
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

    conn.send_frame(&response)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;

    Ok(())
}

async fn handle_agent_inference(
    conn: &mut NexoConnection,
    request_id: &str,
    params: serde_json::Value,
    inference: &InferenceClients,
    prefill_cache: &mut HashMap<String, String>,
) -> utl_helpers::Result {
    let mut messages: Vec<serde_json::Value> = params
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let tools: Vec<serde_json::Value> = params
        .get("tools")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let prefill_sha = params
        .get("prefill_sha")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Resolve prefill by SHA, prepending as system message if present
    if let Some(sha) = &prefill_sha {
        let content = if let Some(cached) = prefill_cache.get(sha.as_str()) {
            cached.clone()
        } else {
            match fetch_prefill_from_gateway(conn, sha).await {
                Ok(content) => {
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

    // Run inference
    let response_payload = if tools.is_empty() {
        let req = ChatRequest {
            messages: chat_messages,
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
        };
        match inference.chat(req).await {
            Ok(resp) => serde_json::json!({ "content": resp.text }),
            Err(e) => {
                tracing::error!("Chat inference failed: {e}");
                let err = Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "inference_error".into(),
                        message: e.to_string(),
                    },
                );
                conn.send_frame(&err)
                    .await
                    .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;
                return Ok(());
            }
        }
    } else {
        let req = ToolCallRequest {
            messages: chat_messages,
            tools,
            max_tokens: 2048,
            temperature: 0.0,
        };
        match inference.tool_call(req).await {
            Ok(resp) => {
                if resp.tool_calls.is_empty() {
                    serde_json::json!({ "content": resp.reasoning.unwrap_or_default() })
                } else {
                    let calls: Vec<serde_json::Value> = resp
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": Frame::new_id(),
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                            })
                        })
                        .collect();
                    serde_json::json!({ "tool_calls": calls })
                }
            }
            Err(e) => {
                tracing::error!("Tool-call inference failed: {e}");
                let err = Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "inference_error".into(),
                        message: e.to_string(),
                    },
                );
                conn.send_frame(&err)
                    .await
                    .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;
                return Ok(());
            }
        }
    };

    let response = Frame::ok_response(request_id, &response_payload).unwrap_or_else(|e| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: e.to_string(),
            },
        )
    });

    conn.send_frame(&response)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;

    Ok(())
}

async fn handle_model_load(
    conn: &mut NexoConnection,
    request_id: &str,
    params: serde_json::Value,
    inference: &InferenceClients,
    currently_loaded: &mut Option<String>,
    available_models: &[String],
) -> utl_helpers::Result {
    let model_id = params
        .get("modelId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    tracing::info!("Loading model '{model_id}'");
    let (loaded, error) = match inference.load_model(&model_id).await {
        Ok(()) => {
            *currently_loaded = Some(model_id.clone());
            (true, None)
        }
        Err(e) => {
            tracing::error!("Failed to load model '{model_id}': {e}");
            (false, Some(e.to_string()))
        }
    };

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

    conn.send_frame(&response)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;

    // Push ModelStatus to gateway so it can update its state
    if loaded {
        push_model_status(conn, Some(model_id), available_models).await;
    }

    Ok(())
}

async fn handle_model_unload(
    conn: &mut NexoConnection,
    request_id: &str,
    params: serde_json::Value,
    inference: &InferenceClients,
    currently_loaded: &mut Option<String>,
    available_models: &[String],
) -> utl_helpers::Result {
    let model_id = params
        .get("modelId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    tracing::info!("Unloading model '{model_id}'");
    let unloaded = match inference.unload_model(&model_id).await {
        Ok(()) => {
            if currently_loaded.as_deref() == Some(model_id.as_str()) {
                *currently_loaded = None;
            }
            true
        }
        Err(e) => {
            tracing::warn!("Unload of model '{model_id}' failed (non-fatal): {e}");
            false
        }
    };

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

    conn.send_frame(&response)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send error: {e}")))?;

    push_model_status(conn, None, available_models).await;

    Ok(())
}

/// Push the node's current loaded model state to the gateway.
async fn push_model_status(
    conn: &mut NexoConnection,
    loaded_model_id: Option<String>,
    available_models: &[String],
) {
    let status = ModelStatusParams {
        loaded_model_id,
        available_models: available_models.to_vec(),
    };
    if let Ok(frame) = Frame::request(Method::ModelStatus, &status) {
        let _ = conn.send_frame(&frame).await;
    }
}

/// Fetch prefill content from the gateway by SHA.
/// Sends a PrefillFetch request and waits for the response (up to 10s).
/// Non-matching frames received while waiting are logged and discarded.
async fn fetch_prefill_from_gateway(
    conn: &mut NexoConnection,
    prefill_sha: &str,
) -> anyhow::Result<String> {
    let fetch_id = Frame::new_id();
    let fetch_frame = Frame::Request {
        id: fetch_id.clone(),
        method: Method::PrefillFetch,
        params: serde_json::json!({ "prefillSha": prefill_sha }),
    };
    conn.send_frame(&fetch_frame)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to send PrefillFetch: {e}"))?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            anyhow::bail!("PrefillFetch timed out for '{prefill_sha}'");
        }

        let frame = tokio::time::timeout(remaining, conn.recv_frame())
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
                tracing::debug!("Discarding non-prefill frame while awaiting PrefillFetch: {other:?}");
            }
        }
    }
}
