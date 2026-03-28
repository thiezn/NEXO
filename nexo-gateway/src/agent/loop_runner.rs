use crate::server::state::SharedState;
use nexo_ws_schema::{
    AgentEventPayload, AgentStatus, EventKind, Frame, Method, ToolsExecuteParams,
    ToolsExecuteResponse,
};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

/// Maximum number of tool-call iterations before forcing a stop.
const MAX_ITERATIONS: usize = 20;

/// Run one complete agent loop for a given run.
pub async fn run(
    run_id: &str,
    session_id: &str,
    prompt: &str,
    context: Option<&serde_json::Value>,
    _peer_id: &str,
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
) {
    // 1. Persist user message
    if let Err(e) =
        super::session::insert_message(db, session_id, Some(run_id), "user", prompt, None, None)
            .await
    {
        tracing::error!("Failed to persist user message: {e}");
        emit_event(event_tx, run_id, session_id, AgentStatus::Failed, None, None);
        let _ = super::session::finish_run(db, run_id, AgentStatus::Failed, Some(&e.to_string())).await;
        return;
    }

    // If extra context was provided, persist it as a system message
    if let Some(ctx) = context {
        let ctx_str = serde_json::to_string(ctx).unwrap_or_default();
        let _ = super::session::insert_message(
            db,
            session_id,
            Some(run_id),
            "system",
            &ctx_str,
            None,
            None,
        )
        .await;
    }

    // 2. Reap expired locks once per run (avoids per-acquire overhead)
    let _ = super::locks::reap_expired(db).await;

    // 3. Enter the inference loop
    for iteration in 0..MAX_ITERATIONS {
        tracing::debug!("Agent loop iteration {iteration} for run {run_id}");

        // Emit thinking event
        emit_event(
            event_tx,
            run_id,
            session_id,
            AgentStatus::Thinking,
            None,
            None,
        );

        // 3. Assemble context
        let messages = match super::context::assemble(db, session_id).await {
            Ok(m) => m,
            Err(e) => {
                fail(event_tx, db, run_id, session_id, &e.to_string()).await;
                return;
            }
        };

        // 4. Get available tools
        let tool_entries = {
            let state_read = state.read().await;
            state_read.all_tool_entries()
        };
        let tool_desc = super::context::build_tool_descriptions(&tool_entries);

        // 5. Find an LLM-capable node and run inference
        let inference_result = run_inference(
            run_id, &messages, &tool_desc, db, state, event_tx,
        )
        .await;

        match inference_result {
            InferenceOutcome::Reply(text) => {
                // Persist assistant message
                let _ = super::session::insert_message(
                    db,
                    session_id,
                    Some(run_id),
                    "assistant",
                    &text,
                    None,
                    None,
                )
                .await;

                // Stream content event
                emit_event(
                    event_tx,
                    run_id,
                    session_id,
                    AgentStatus::Streaming,
                    Some(&text),
                    None,
                );

                // Mark completed
                emit_event(
                    event_tx,
                    run_id,
                    session_id,
                    AgentStatus::Completed,
                    None,
                    None,
                );
                let _ =
                    super::session::finish_run(db, run_id, AgentStatus::Completed, Some(&text)).await;
                super::locks::release_all_for_run(db, run_id).await.ok();
                return;
            }
            InferenceOutcome::ToolCalls(calls) => {
                for call in &calls {
                    // Emit tool_call event
                    emit_event(
                        event_tx,
                        run_id,
                        session_id,
                        AgentStatus::ToolCall,
                        None,
                        Some(&call.name),
                    );

                    // Attempt capability lock
                    let capability = tool_capability(&call.name);
                    match super::locks::acquire(db, &capability, run_id).await {
                        Ok(true) => {}
                        Ok(false) => {
                            tracing::warn!(
                                "Capability '{capability}' locked, skipping tool '{}'",
                                call.name
                            );
                            let _ = super::session::insert_message(
                                db,
                                session_id,
                                Some(run_id),
                                "tool",
                                &format!("Error: capability '{capability}' is busy"),
                                Some(&call.id),
                                Some(&call.name),
                            )
                            .await;
                            continue;
                        }
                        Err(e) => {
                            tracing::error!("Lock acquire failed: {e}");
                            continue;
                        }
                    }

                    // Execute tool
                    let tool_result = execute_tool(&call.name, &call.arguments, state).await;

                    // Release lock
                    super::locks::release(db, &capability).await.ok();

                    // Persist tool result
                    let output = match &tool_result {
                        Ok(resp) if resp.success => resp.output.clone(),
                        Ok(resp) => {
                            format!(
                                "Error: {}",
                                resp.error.as_deref().unwrap_or("unknown error")
                            )
                        }
                        Err(e) => format!("Error: {e}"),
                    };

                    let _ = super::session::insert_message(
                        db,
                        session_id,
                        Some(run_id),
                        "tool",
                        &output,
                        Some(&call.id),
                        Some(&call.name),
                    )
                    .await;
                }

                // Continue loop -- tool results are now in context for next inference
                continue;
            }
            InferenceOutcome::Error(err) => {
                fail(event_tx, db, run_id, session_id, &err).await;
                return;
            }
            InferenceOutcome::NoLlmAvailable => {
                fail(
                    event_tx,
                    db,
                    run_id,
                    session_id,
                    "No LLM-capable node is connected",
                )
                .await;
                return;
            }
        }
    }

    // Exceeded max iterations
    fail(
        event_tx,
        db,
        run_id,
        session_id,
        &format!("Agent loop exceeded {MAX_ITERATIONS} iterations"),
    )
    .await;
}

// ── Internal types ──────────────────────────────────────────

enum InferenceOutcome {
    Reply(String),
    ToolCalls(Vec<ToolCallInfo>),
    Error(String),
    NoLlmAvailable,
}

struct ToolCallInfo {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

// ── Inference execution ─────────────────────────────────────

/// Forward an inference request to an LLM-capable node and parse the response.
async fn run_inference(
    run_id: &str,
    messages: &[super::context::ContextMessage],
    tool_descriptions: &str,
    _db: &SqlitePool,
    state: &SharedState,
    _event_tx: &broadcast::Sender<Frame>,
) -> InferenceOutcome {
    // Build a chat-style payload for the node's inference endpoint
    let chat_messages: Vec<serde_json::Value> = std::iter::once(serde_json::json!({
        "role": "system",
        "content": format!("You are a helpful assistant.\n\n{tool_descriptions}")
    }))
    .chain(messages.iter().map(|m| {
        let mut msg = serde_json::json!({
            "role": m.role,
            "content": m.content,
        });
        if let Some(ref tc_id) = m.tool_call_id {
            msg["tool_call_id"] = serde_json::Value::String(tc_id.clone());
        }
        if let Some(ref tn) = m.tool_name {
            msg["tool_name"] = serde_json::Value::String(tn.clone());
        }
        msg
    }))
    .collect();

    let inference_payload = serde_json::json!({
        "messages": chat_messages,
        "run_id": run_id,
    });

    // Find an LLM-capable node
    let (node_sender, forwarded_id) = {
        let state_read = state.read().await;
        let llm_peer = state_read
            .peers
            .values()
            .find(|p| {
                p.role == nexo_ws_schema::Role::Node
                    && p.capabilities.iter().any(|c| c == "llm" || c == "inference")
            });

        let Some(peer) = llm_peer else {
            return InferenceOutcome::NoLlmAvailable;
        };

        let Some(sender) = state_read.peer_senders.get(&peer.id) else {
            return InferenceOutcome::NoLlmAvailable;
        };

        (sender.clone(), Frame::new_id())
    };

    // Build forwarded request
    let forwarded_frame = Frame::Request {
        id: forwarded_id.clone(),
        method: Method::Agent,
        params: inference_payload,
    };

    // Register oneshot for response
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    {
        let mut state_write = state.write().await;
        state_write
            .pending_requests
            .insert(forwarded_id.clone(), response_tx);
    }

    // Send to node
    if node_sender.send(forwarded_frame).await.is_err() {
        let mut state_write = state.write().await;
        state_write.pending_requests.remove(&forwarded_id);
        return InferenceOutcome::Error("Failed to send inference request to node".into());
    }

    // Await response with 120s timeout (inference can be slow)
    match tokio::time::timeout(std::time::Duration::from_secs(120), response_rx).await {
        Ok(Ok(Frame::Response {
            ok: true, payload, ..
        })) => parse_inference_response(payload),
        Ok(Ok(Frame::Response {
            ok: false, error, ..
        })) => InferenceOutcome::Error(
            error
                .map(|e| e.message)
                .unwrap_or_else(|| "Inference failed".into()),
        ),
        Ok(Ok(_)) => InferenceOutcome::Error("Unexpected frame type from node".into()),
        Ok(Err(_)) => InferenceOutcome::Error("Node disconnected during inference".into()),
        Err(_) => {
            let mut state_write = state.write().await;
            state_write.pending_requests.remove(&forwarded_id);
            InferenceOutcome::Error("Inference timed out (120s)".into())
        }
    }
}

/// Parse the inference response to determine if it's a reply or tool calls.
fn parse_inference_response(payload: Option<serde_json::Value>) -> InferenceOutcome {
    let Some(payload) = payload else {
        return InferenceOutcome::Error("Empty inference response".into());
    };

    // Check for tool_calls in the response
    if let Some(tool_calls) = payload.get("tool_calls").and_then(|v| v.as_array()) {
        let calls: Vec<ToolCallInfo> = tool_calls
            .iter()
            .filter_map(|tc| {
                Some(ToolCallInfo {
                    id: tc.get("id")?.as_str()?.to_string(),
                    name: tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    arguments: tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                })
            })
            .collect();

        if !calls.is_empty() {
            return InferenceOutcome::ToolCalls(calls);
        }
    }

    // Plain text reply
    let content = payload
        .get("content")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("output").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();

    InferenceOutcome::Reply(content)
}

// ── Tool execution ──────────────────────────────────────────

/// Execute a tool by forwarding to the owning node via the gateway's directed channel.
async fn execute_tool(
    tool_name: &str,
    args: &serde_json::Value,
    state: &SharedState,
) -> Result<ToolsExecuteResponse, String> {
    let (node_sender, forwarded_id) = {
        let state_read = state.read().await;
        let tool = state_read
            .find_tool(tool_name)
            .ok_or_else(|| format!("Tool '{tool_name}' not found"))?;
        let sender = state_read
            .peer_senders
            .get(&tool.peer_id)
            .cloned()
            .ok_or_else(|| format!("Node hosting tool '{tool_name}' is disconnected"))?;
        (sender, Frame::new_id())
    };

    let exec_params = ToolsExecuteParams {
        tool: tool_name.to_string(),
        args: args.clone(),
        idempotency_key: Frame::new_id(),
    };

    let forwarded_frame = match Frame::request(Method::ToolsExecute, &exec_params) {
        Ok(mut f) => {
            if let Frame::Request { ref mut id, .. } = f {
                *id = forwarded_id.clone();
            }
            f
        }
        Err(e) => return Err(format!("Failed to build tool request: {e}")),
    };

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    {
        let mut state_write = state.write().await;
        state_write
            .pending_requests
            .insert(forwarded_id.clone(), response_tx);
    }

    if node_sender.send(forwarded_frame).await.is_err() {
        let mut state_write = state.write().await;
        state_write.pending_requests.remove(&forwarded_id);
        return Err("Failed to send tool request to node".into());
    }

    match tokio::time::timeout(std::time::Duration::from_secs(30), response_rx).await {
        Ok(Ok(Frame::Response {
            ok: true, payload, ..
        })) => {
            let resp: ToolsExecuteResponse = payload
                .and_then(|p| serde_json::from_value(p).ok())
                .unwrap_or(ToolsExecuteResponse {
                    success: false,
                    output: String::new(),
                    error: Some("Invalid tool response".into()),
                });
            Ok(resp)
        }
        Ok(Ok(Frame::Response { error, .. })) => Ok(ToolsExecuteResponse {
            success: false,
            output: String::new(),
            error: error.map(|e| e.message),
        }),
        Ok(Ok(_)) => Err("Unexpected frame type from node".into()),
        Ok(Err(_)) => Err("Node disconnected during tool execution".into()),
        Err(_) => {
            let mut state_write = state.write().await;
            state_write.pending_requests.remove(&forwarded_id);
            Err("Tool execution timed out (30s)".into())
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────

/// Derive a capability name from a tool name (e.g. "echo.run" -> "echo").
fn tool_capability(tool_name: &str) -> String {
    tool_name
        .split('.')
        .next()
        .unwrap_or(tool_name)
        .to_string()
}

fn emit_event(
    event_tx: &broadcast::Sender<Frame>,
    run_id: &str,
    session_id: &str,
    status: AgentStatus,
    content: Option<&str>,
    tool_name: Option<&str>,
) {
    let payload = AgentEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status,
        content: content.map(String::from),
        tool_name: tool_name.map(String::from),
        tool_call_id: None,
        error: None,
    };
    if let Ok(frame) = Frame::event(EventKind::Agent, &payload) {
        let _ = event_tx.send(frame);
    }
}

async fn fail(
    event_tx: &broadcast::Sender<Frame>,
    db: &SqlitePool,
    run_id: &str,
    session_id: &str,
    error: &str,
) {
    tracing::error!("Agent run {run_id} failed: {error}");
    let payload = AgentEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status: AgentStatus::Failed,
        content: None,
        tool_name: None,
        tool_call_id: None,
        error: Some(error.to_string()),
    };
    if let Ok(frame) = Frame::event(EventKind::Agent, &payload) {
        let _ = event_tx.send(frame);
    }
    let _ = super::session::finish_run(db, run_id, AgentStatus::Failed, Some(error)).await;
    super::locks::release_all_for_run(db, run_id).await.ok();
}
