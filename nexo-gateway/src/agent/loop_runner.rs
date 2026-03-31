use crate::server::state::SharedState;
use nexo_ws_schema::{
    AgentEventPayload, AgentStatus, EventKind, Frame, Method, ModelLoadParams, ModelLoadResponse,
    ModelUnloadParams, ToolEntry, ToolsExecuteParams, ToolsExecuteResponse,
};
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};

/// Maximum number of tool-call iterations before forcing a stop.
const MAX_ITERATIONS: usize = 20;
/// Timeout for model load operations (models can be large).
const MODEL_LOAD_TIMEOUT_SECS: u64 = 300;
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

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
    model_id: Option<&str>,
    prefill_collection_id: Option<&str>,
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

    // 2b. Load SOUL.md (always prepended) and optionally resolve prefill collection.
    // Clone the Arc and drop the lock before doing blocking git I/O.
    let git = state.read().await.git_storage.clone();
    let (soul_content, prefill_content) = if let Some(ref git) = git {
        let g = git.clone();
        let cid = prefill_collection_id.map(String::from);
        tokio::task::spawn_blocking(move || {
            let soul = g.read_file("SOUL.md").unwrap_or_default();
            let prefill = cid.and_then(|cid| {
                super::prefill::resolve_collection(&g, &cid)
                    .ok()
                    .flatten()
                    .map(|(content, _sha)| content)
            });
            (soul, prefill)
        })
        .await
        .unwrap_or_default()
    } else {
        (String::new(), None)
    };

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

        // 3a. Assemble context
        let messages = match super::context::assemble(db, session_id).await {
            Ok(m) => m,
            Err(e) => {
                fail(event_tx, db, run_id, session_id, &e.to_string()).await;
                return;
            }
        };

        // 3b. Get available tools
        let tool_entries = {
            let state_read = state.read().await;
            state_read.all_tool_entries()
        };
        let tool_desc = super::context::build_tool_descriptions(&tool_entries);

        // 3c. Find an LLM-capable node and run inference
        let inference_result = run_inference(
            run_id,
            &messages,
            &tool_desc,
            &tool_entries,
            model_id,
            &soul_content,
            prefill_content.as_deref(),
            state,
            event_tx,
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
                // Queue the run for later processing instead of failing
                queue_run(db, run_id, prompt, context, _peer_id, model_id).await;
                emit_queued_event(event_tx, run_id, session_id);
                // Don't call finish_run(failed) — the run stays in 'queued' status
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

// ── Model loading ────────────────────────────────────────────

/// Ensure the given model is loaded on a node, loading it if necessary.
/// Returns (node_sender, forwarded_id) on success, or an InferenceOutcome error.
async fn ensure_model_loaded(
    model_id: &str,
    state: &SharedState,
) -> Result<mpsc::Sender<Frame>, InferenceOutcome> {
    // Step 1: Check if any node already has the model loaded in VRAM
    {
        let state_read = state.read().await;
        if let Some((_, sender)) = state_read.find_loaded_llm_peer(model_id) {
            return Ok(sender);
        }
    }

    // Step 2: Find a node that has the model on disk
    let (peer_id, node_sender) = {
        let state_read = state.read().await;
        match state_read.find_capable_peer_for_model(model_id) {
            Some((pid, sender)) => (pid, sender),
            None => return Err(InferenceOutcome::NoLlmAvailable),
        }
    };

    // Step 3: Check if the node has a different model loaded — if so, unload it first
    let currently_loaded = state
        .read()
        .await
        .loaded_models
        .get(&peer_id)
        .and_then(|m| m.clone());

    if let Some(old_model) = currently_loaded {
        tracing::info!("Unloading model '{old_model}' from node {peer_id} before loading '{model_id}'");
        let unload_params = ModelUnloadParams {
            model_id: old_model.clone(),
        };
        let unload_fwd_id = Frame::new_id();
        let frame = Frame::Request {
            id: unload_fwd_id.clone(),
            method: Method::ModelUnload,
            params: serde_json::to_value(&unload_params).unwrap_or_default(),
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        state.write().await.pending_requests.insert(unload_fwd_id.clone(), tx);
        if node_sender.send(frame).await.is_err() {
            state.write().await.pending_requests.remove(&unload_fwd_id);
        } else {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(10), rx).await;
        }
        state.write().await.set_loaded_model(&peer_id, None);
    }

    // Step 4: Send ModelLoad to the node
    tracing::info!("Loading model '{model_id}' on node {peer_id}");
    let load_params = ModelLoadParams {
        model_id: model_id.to_string(),
    };
    let load_fwd_id = Frame::new_id();
    let load_frame = Frame::Request {
        id: load_fwd_id.clone(),
        method: Method::ModelLoad,
        params: serde_json::to_value(&load_params).unwrap_or_default(),
    };

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    state.write().await.pending_requests.insert(load_fwd_id.clone(), response_tx);

    if node_sender.send(load_frame).await.is_err() {
        state.write().await.pending_requests.remove(&load_fwd_id);
        return Err(InferenceOutcome::Error(format!(
            "Failed to send ModelLoad request to node {peer_id}"
        )));
    }

    // Step 5: Await load response (300s — models can be large)
    match tokio::time::timeout(
        std::time::Duration::from_secs(MODEL_LOAD_TIMEOUT_SECS),
        response_rx,
    )
    .await
    {
        Ok(Ok(Frame::Response { ok: true, payload, .. })) => {
            let loaded = payload
                .as_ref()
                .and_then(|p| serde_json::from_value::<ModelLoadResponse>(p.clone()).ok())
                .map(|r| r.loaded)
                .unwrap_or(true);

            if loaded {
                state.write().await.set_loaded_model(&peer_id, Some(model_id.to_string()));
                tracing::info!("Model '{model_id}' loaded on node {peer_id}");
                Ok(node_sender)
            } else {
                Err(InferenceOutcome::Error(format!(
                    "Node {peer_id} failed to load model '{model_id}'"
                )))
            }
        }
        Ok(Ok(Frame::Response { ok: false, error, .. })) => Err(InferenceOutcome::Error(
            error
                .map(|e| format!("ModelLoad error: {}", e.message))
                .unwrap_or_else(|| format!("ModelLoad failed on node {peer_id}")),
        )),
        Ok(Ok(_)) => Err(InferenceOutcome::Error("Unexpected frame type from node during model load".into())),
        Ok(Err(_)) => Err(InferenceOutcome::Error("Node disconnected during model load".into())),
        Err(_) => {
            state.write().await.pending_requests.remove(&load_fwd_id);
            Err(InferenceOutcome::Error(format!(
                "Model load timed out after {MODEL_LOAD_TIMEOUT_SECS}s"
            )))
        }
    }
}

// ── Inference execution ─────────────────────────────────────

/// Forward an inference request to an LLM-capable node and parse the response.
async fn run_inference(
    run_id: &str,
    messages: &[super::context::ContextMessage],
    tool_descriptions: &str,
    tool_entries: &[ToolEntry],
    model_id: Option<&str>,
    soul_content: &str,
    prefill_content: Option<&str>,
    state: &SharedState,
    _event_tx: &broadcast::Sender<Frame>,
) -> InferenceOutcome {
    // Build system prompt from SOUL.md, prefill, and tool descriptions
    let mut system_parts = Vec::new();
    if !soul_content.is_empty() {
        system_parts.push(soul_content.to_string());
    }
    if let Some(prefill) = prefill_content {
        if !prefill.is_empty() {
            system_parts.push(prefill.to_string());
        }
    }
    if !tool_descriptions.is_empty() {
        system_parts.push(tool_descriptions.to_string());
    }
    let system_prompt = if system_parts.is_empty() {
        DEFAULT_SYSTEM_PROMPT.to_string()
    } else {
        system_parts.join("\n\n")
    };

    // Build a chat-style payload for the node's inference endpoint
    let chat_messages: Vec<serde_json::Value> = std::iter::once(serde_json::json!({
        "role": "system",
        "content": system_prompt
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

    // Build structured tools in OpenAI function-calling format
    let tools_json: Vec<serde_json::Value> = tool_entries
        .iter()
        .filter(|t| t.available)
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters.clone().unwrap_or(serde_json::json!({"type": "object"}))
                }
            })
        })
        .collect();

    let mut inference_payload = serde_json::json!({
        "messages": chat_messages,
        "run_id": run_id,
        "tools": tools_json,
    });

    if let Some(mid) = model_id {
        inference_payload["model_id"] = serde_json::Value::String(mid.to_string());
    }

    // Find an LLM-capable node (with model loading if model_id is specified)
    let node_sender = match model_id {
        Some(mid) => match ensure_model_loaded(mid, state).await {
            Ok(sender) => sender,
            Err(outcome) => return outcome,
        },
        None => {
            let state_read = state.read().await;
            match state_read.find_any_llm_peer() {
                Some((_, sender)) => sender,
                None => return InferenceOutcome::NoLlmAvailable,
            }
        }
    };
    let forwarded_id = Frame::new_id();

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

/// Execute a tool — first check gateway-native tools, then forward to a node.
async fn execute_tool(
    tool_name: &str,
    args: &serde_json::Value,
    state: &SharedState,
) -> Result<ToolsExecuteResponse, String> {
    // Check gateway-native tools first (notes, etc.)
    let gateway_tool = {
        let state_read = state.read().await;
        state_read.gateway_tools.get_tool(tool_name).cloned()
    };
    if let Some(tool) = gateway_tool {
        return match tool.execute(args.clone()).await {
            Ok(tr) => Ok(ToolsExecuteResponse {
                success: tr.success,
                output: tr.output,
                error: tr.error,
            }),
            Err(e) => Err(format!("Gateway tool error: {e}")),
        };
    }

    // Otherwise, forward to owning node
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

// ── Queue helpers ────────────────────────────────────────────

/// Persist a run as 'queued' when no LLM is available.
async fn queue_run(
    db: &SqlitePool,
    run_id: &str,
    prompt: &str,
    context: Option<&serde_json::Value>,
    peer_id: &str,
    model_id: Option<&str>,
) {
    let context_json = context.and_then(|c| serde_json::to_string(c).ok());
    if let Err(e) = sqlx::query(
        "UPDATE agent_runs
         SET status = 'queued', queued_at = datetime('now'),
             queued_prompt = ?, queued_context = ?, queued_peer_id = ?, model_id = ?
         WHERE id = ?",
    )
    .bind(prompt)
    .bind(context_json.as_deref())
    .bind(peer_id)
    .bind(model_id)
    .bind(run_id)
    .execute(db)
    .await
    {
        tracing::error!("Failed to queue run {run_id}: {e}");
    } else {
        tracing::info!("Run {run_id} queued (no LLM available)");
    }
}

/// Emit a 'queued' event to inform the client their request is pending.
fn emit_queued_event(event_tx: &broadcast::Sender<Frame>, run_id: &str, session_id: &str) {
    let payload = AgentEventPayload {
        run_id: run_id.to_string(),
        session_id: session_id.to_string(),
        status: AgentStatus::Queued,
        content: Some(
            "No inference node is currently available. Your request has been queued and will be processed as soon as a node becomes available.".to_string()
        ),
        tool_name: None,
        tool_call_id: None,
        error: None,
    };
    if let Ok(frame) = Frame::event(EventKind::Agent, &payload) {
        let _ = event_tx.send(frame);
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
