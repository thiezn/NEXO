use crate::{agent::persistence, server::state::SharedState, tools};
use nexo_core::{ConversationMessage, ToolCall};
use nexo_ws_schema::{
    Frame, Method, ModelLoadParams, ModelLoadResponse, ModelUnloadParams, RunRoundRequest,
    RunRoundResponse, RunStatus, ToolEntry,
};
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};

use super::{context_manager::ContextManager, events};

/// Maximum number of rounds before the engine stops a run.
const MAX_ITERATIONS: usize = 20;
/// Timeout for model load operations.
const MODEL_LOAD_TIMEOUT_SECS: u64 = 300;
pub(super) const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

// TODO: The think tokens and thinking block are hardcoded here but they relate specifically to gemma4.
// We should pull this from model definition instead.
pub(super) const THINK_TOKEN: &str = "<|think|>";
const THINKING_BLOCK_OPEN: &str = "<|channel>thought\n";
const THINKING_BLOCK_CLOSE: &str = "<channel|>";

/// Return whether a run has already been cancelled.
///
/// Database lookup failures are logged and treated as not-cancelled so the loop
/// can fail through its normal execution path.
async fn run_cancelled(db: &SqlitePool, run_id: &str) -> bool {
    match persistence::is_run_cancelled(db, run_id).await {
        Ok(cancelled) => cancelled,
        Err(error) => {
            tracing::error!("Failed to load run status for {run_id}: {error}");
            false
        }
    }
}

/// Start a run from a fresh user request.
#[expect(clippy::too_many_arguments)]
pub async fn start_run(
    run_id: &str,
    session_id: &str,
    input: &str,
    instructions: Option<&serde_json::Value>,
    peer_id: &str,
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
    model_id: Option<&str>,
    prompt_collection_id: Option<&str>,
    thinking: bool,
) {
    if let Err(error) = persistence::insert_conversation_entry(
        db,
        session_id,
        Some(run_id),
        None,
        "user",
        input,
        persistence::ENTRY_USER_INPUT,
        None,
        None,
    )
    .await
    {
        events::fail_run(event_tx, db, run_id, session_id, &error.to_string()).await;
        return;
    }

    if let Some(extra_instructions) = instructions {
        let instructions_string = serde_json::to_string(extra_instructions).unwrap_or_default();
        let _ = persistence::insert_conversation_entry(
            db,
            session_id,
            Some(run_id),
            None,
            "system",
            &instructions_string,
            persistence::ENTRY_INSTRUCTION,
            None,
            None,
        )
        .await;
    }

    run_engine(
        run_id,
        session_id,
        peer_id,
        db,
        state,
        event_tx,
        model_id,
        prompt_collection_id,
        thinking,
    )
    .await;
}

/// Resume a queued run without replaying its original conversation input.
#[expect(clippy::too_many_arguments)]
pub async fn resume_run(
    run_id: &str,
    session_id: &str,
    peer_id: &str,
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
    model_id: Option<&str>,
    prompt_collection_id: Option<&str>,
    thinking: bool,
) {
    run_engine(
        run_id,
        session_id,
        peer_id,
        db,
        state,
        event_tx,
        model_id,
        prompt_collection_id,
        thinking,
    )
    .await;
}

/// Drive the round-based loop for a run until it completes, fails, or queues.
#[expect(clippy::too_many_arguments)]
async fn run_engine(
    run_id: &str,
    session_id: &str,
    _peer_id: &str,
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
    model_id: Option<&str>,
    prompt_collection_id: Option<&str>,
    thinking: bool,
) {
    let _ = crate::agent::locks::reap_expired(db).await;

    let context_manager = ContextManager::new(state, prompt_collection_id, thinking).await;

    let starting_round_index = match persistence::next_round_index(db, run_id).await {
        Ok(index) => index,
        Err(error) => {
            events::fail_run(event_tx, db, run_id, session_id, &error.to_string()).await;
            return;
        }
    };

    for round_index in starting_round_index..=MAX_ITERATIONS {
        if run_cancelled(db, run_id).await {
            crate::agent::locks::release_all_for_run(db, run_id)
                .await
                .ok();
            return;
        }

        let round_id = match persistence::create_round(db, run_id, round_index, model_id).await {
            Ok(round_id) => round_id,
            Err(error) => {
                events::fail_run(event_tx, db, run_id, session_id, &error.to_string()).await;
                return;
            }
        };

        events::emit_status(
            event_tx,
            run_id,
            session_id,
            RunStatus::Thinking,
            None,
            None,
        );

        let tool_entries = {
            let state_read = state.read().await;
            state_read.all_tool_entries()
        };

        let prepared_context = match context_manager
            .prepare_round_context(db, session_id, &tool_entries)
            .await
        {
            Ok(context) => context,
            Err(error) => {
                let _ = persistence::finish_round(db, &round_id, "failed", None, None).await;
                events::fail_run(event_tx, db, run_id, session_id, &error.to_string()).await;
                return;
            }
        };

        tracing::info!(
            "Run {run_id} entering round {} (round_id={round_id}, messages={}, tools={}, model={:?}, thinking={thinking})",
            round_index,
            prepared_context.persisted_message_count,
            tool_entries.iter().filter(|tool| tool.available).count(),
            model_id,
        );

        let inference = run_inference(
            run_id,
            &round_id,
            prepared_context.round_messages,
            &tool_entries,
            model_id,
            state,
            session_id,
        )
        .await;

        if run_cancelled(db, run_id).await {
            let _ = persistence::finish_round(db, &round_id, "cancelled", None, None).await;
            crate::agent::locks::release_all_for_run(db, run_id)
                .await
                .ok();
            return;
        }

        match inference {
            InferenceOutcome::Reply(reply) => {
                let (visible_text, thinking_content) = if thinking {
                    strip_thinking_content(&reply.raw_text)
                } else {
                    (reply.raw_text, None)
                };
                tracing::info!(
                    "Run {run_id} round {round_id} completed with assistant reply from {} (chars={}, rationale_chars={})",
                    reply.selected_peer_id,
                    visible_text.len(),
                    reply.rationale.as_deref().map_or(0, str::len),
                );

                let _ = persistence::insert_conversation_entry(
                    db,
                    session_id,
                    Some(run_id),
                    Some(&round_id),
                    "assistant",
                    &visible_text,
                    persistence::ENTRY_ASSISTANT_RESPONSE,
                    None,
                    None,
                )
                .await;
                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    "completed",
                    reply.rationale.as_deref(),
                    Some(&reply.selected_peer_id),
                )
                .await;

                events::emit_status_with_thinking(
                    event_tx,
                    run_id,
                    session_id,
                    RunStatus::Streaming,
                    Some(&visible_text),
                    None,
                    thinking_content.as_deref(),
                );
                events::emit_status(
                    event_tx,
                    run_id,
                    session_id,
                    RunStatus::Completed,
                    None,
                    None,
                );

                let _ =
                    persistence::finish_run(db, run_id, RunStatus::Completed, Some(&visible_text))
                        .await;
                crate::agent::locks::release_all_for_run(db, run_id)
                    .await
                    .ok();
                return;
            }
            InferenceOutcome::ToolCalls(outcome) => {
                tracing::info!(
                    "Run {run_id} round {round_id} requested {} tool call(s) from {}",
                    outcome.calls.len(),
                    outcome.selected_peer_id,
                );
                let assistant_tool_history = serialize_tool_calls_for_history(&outcome.calls);
                let _ = persistence::insert_conversation_entry(
                    db,
                    session_id,
                    Some(run_id),
                    Some(&round_id),
                    "assistant",
                    &assistant_tool_history,
                    persistence::ENTRY_TOOL_CALL_INTENT,
                    None,
                    None,
                )
                .await;

                for call in &outcome.calls {
                    if run_cancelled(db, run_id).await {
                        let _ = persistence::finish_round(
                            db,
                            &round_id,
                            "cancelled",
                            None,
                            Some(&outcome.selected_peer_id),
                        )
                        .await;
                        crate::agent::locks::release_all_for_run(db, run_id)
                            .await
                            .ok();
                        return;
                    }

                    let trace_id = persistence::create_tool_trace(
                        db,
                        run_id,
                        &round_id,
                        call.id.as_str(),
                        &call.name,
                        &call.arguments,
                    )
                    .await
                    .ok();

                    tracing::info!(
                        "Run {run_id} round {round_id} invoking tool {} (call_id={})",
                        call.name,
                        call.id,
                    );
                    events::emit_tool_started(
                        event_tx,
                        run_id,
                        session_id,
                        &call.name,
                        call.id.as_str(),
                    );

                    let capability = tools::tool_capability(&call.name);
                    match crate::agent::locks::acquire(db, &capability, run_id).await {
                        Ok(true) => {}
                        Ok(false) => {
                            let output = format!("Error: capability '{capability}' is busy");
                            if let Some(trace_id) = trace_id.as_deref() {
                                let _ = persistence::finish_tool_trace(
                                    db,
                                    trace_id,
                                    false,
                                    None,
                                    Some(&output),
                                )
                                .await;
                            }
                            let _ = persistence::insert_conversation_entry(
                                db,
                                session_id,
                                Some(run_id),
                                Some(&round_id),
                                "tool",
                                &output,
                                persistence::ENTRY_TOOL_RESULT,
                                Some(call.id.as_str()),
                                Some(&call.name),
                            )
                            .await;
                            events::emit_tool_result(
                                event_tx,
                                run_id,
                                session_id,
                                &call.name,
                                call.id.as_str(),
                                &output,
                            );
                            continue;
                        }
                        Err(error) => {
                            if let Some(trace_id) = trace_id.as_deref() {
                                let _ = persistence::finish_tool_trace(
                                    db,
                                    trace_id,
                                    false,
                                    None,
                                    Some(&error.to_string()),
                                )
                                .await;
                            }
                            tracing::error!("Lock acquire failed: {error}");
                            continue;
                        }
                    }

                    let tool_result = tools::execute_tool(call.clone(), state).await;
                    crate::agent::locks::release(db, &capability).await.ok();

                    let (success, output, error_message) = match &tool_result {
                        Ok(response) if response.success => (true, response.output.clone(), None),
                        Ok(response) => {
                            let error_message = response
                                .error
                                .clone()
                                .unwrap_or_else(|| "unknown error".into());
                            (
                                false,
                                format!("Error: {error_message}"),
                                Some(error_message),
                            )
                        }
                        Err(error) => (false, format!("Error: {error}"), Some(error.clone())),
                    };

                    if let Some(trace_id) = trace_id.as_deref() {
                        let _ = persistence::finish_tool_trace(
                            db,
                            trace_id,
                            success,
                            Some(&output),
                            error_message.as_deref(),
                        )
                        .await;
                    }

                    let _ = persistence::insert_conversation_entry(
                        db,
                        session_id,
                        Some(run_id),
                        Some(&round_id),
                        "tool",
                        &output,
                        persistence::ENTRY_TOOL_RESULT,
                        Some(call.id.as_str()),
                        Some(&call.name),
                    )
                    .await;
                    tracing::info!(
                        "Run {run_id} round {round_id} tool {} finished (call_id={}, success={}, output_chars={})",
                        call.name,
                        call.id,
                        success,
                        output.len(),
                    );
                    events::emit_tool_result(
                        event_tx,
                        run_id,
                        session_id,
                        &call.name,
                        call.id.as_str(),
                        &output,
                    );
                }

                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    "completed",
                    outcome.rationale.as_deref(),
                    Some(&outcome.selected_peer_id),
                )
                .await;
                continue;
            }
            InferenceOutcome::Error(error) => {
                tracing::info!("Run {run_id} round {round_id} failed: {error}");
                let _ = persistence::finish_round(db, &round_id, "failed", None, None).await;
                events::fail_run(event_tx, db, run_id, session_id, &error).await;
                return;
            }
            InferenceOutcome::NoLlmAvailable => {
                tracing::info!(
                    "Run {run_id} round {round_id} queued (no inference node available)"
                );
                let _ = persistence::finish_round(db, &round_id, "queued", None, None).await;
                persistence::mark_run_queued(db, run_id).await;
                events::emit_queued_event(event_tx, run_id, session_id);
                return;
            }
        }
    }

    events::fail_run(
        event_tx,
        db,
        run_id,
        session_id,
        &format!("Run loop exceeded {MAX_ITERATIONS} iterations"),
    )
    .await;
}

/// Normalized result of one gateway-to-node inference round.
enum InferenceOutcome {
    Reply(ReplyOutcome),
    ToolCalls(ToolCallOutcome),
    Error(String),
    NoLlmAvailable,
}

/// Terminal assistant reply for a round.
struct ReplyOutcome {
    raw_text: String,
    rationale: Option<String>,
    selected_peer_id: String,
}

/// Tool-call plan returned for a round.
struct ToolCallOutcome {
    calls: Vec<ToolCall>,
    rationale: Option<String>,
    selected_peer_id: String,
}

/// Ensure the requested model is loaded and return the selected node.
async fn ensure_model_loaded(
    model_id: &str,
    state: &SharedState,
) -> Result<(String, mpsc::Sender<Frame>), InferenceOutcome> {
    {
        let state_read = state.read().await;
        if let Some((peer_id, sender)) = state_read.find_loaded_llm_peer(model_id) {
            return Ok((peer_id, sender));
        }
    }

    let (peer_id, node_sender) = {
        let state_read = state.read().await;
        match state_read.find_capable_peer_for_model(model_id) {
            Some((peer_id, sender)) => (peer_id, sender),
            None => return Err(InferenceOutcome::NoLlmAvailable),
        }
    };

    let models_to_unload: Vec<String> = state
        .read()
        .await
        .loaded_models
        .get(&peer_id)
        .map(|models| {
            models
                .iter()
                .filter(|model| model.id.as_str() != model_id)
                .map(|model| model.id.to_string())
                .collect()
        })
        .unwrap_or_default();

    for old_model in &models_to_unload {
        let unload_params = ModelUnloadParams {
            model_id: old_model.clone(),
        };
        let unload_request_id = Frame::new_id();
        let frame = Frame::Request {
            id: unload_request_id.clone(),
            method: Method::ModelUnload,
            params: serde_json::to_value(&unload_params).unwrap_or_default(),
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        state
            .write()
            .await
            .pending_requests
            .insert(unload_request_id.clone(), tx);
        if node_sender.send(frame).await.is_err() {
            state
                .write()
                .await
                .pending_requests
                .remove(&unload_request_id);
        } else {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(10), rx).await;
        }
    }

    if !models_to_unload.is_empty() {
        state.write().await.set_loaded_models(&peer_id, Vec::new());
    }

    let load_params = ModelLoadParams {
        model_id: model_id.to_string(),
    };
    let load_request_id = Frame::new_id();
    let frame = Frame::Request {
        id: load_request_id.clone(),
        method: Method::ModelLoad,
        params: serde_json::to_value(&load_params).unwrap_or_default(),
    };

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    state
        .write()
        .await
        .pending_requests
        .insert(load_request_id.clone(), response_tx);

    if node_sender.send(frame).await.is_err() {
        state
            .write()
            .await
            .pending_requests
            .remove(&load_request_id);
        return Err(InferenceOutcome::Error(format!(
            "Failed to send ModelLoad request to node {peer_id}"
        )));
    }

    match tokio::time::timeout(
        std::time::Duration::from_secs(MODEL_LOAD_TIMEOUT_SECS),
        response_rx,
    )
    .await
    {
        Ok(Ok(Frame::Response {
            ok: true, payload, ..
        })) => {
            let loaded = payload
                .as_ref()
                .and_then(|payload| {
                    serde_json::from_value::<ModelLoadResponse>(payload.clone()).ok()
                })
                .map(|response| response.loaded)
                .unwrap_or(true);
            if loaded {
                Ok((peer_id, node_sender))
            } else {
                Err(InferenceOutcome::Error(format!(
                    "Node {peer_id} failed to load model '{model_id}'"
                )))
            }
        }
        Ok(Ok(Frame::Response {
            ok: false, error, ..
        })) => Err(InferenceOutcome::Error(
            error
                .map(|payload| format!("ModelLoad error: {}", payload.message))
                .unwrap_or_else(|| format!("ModelLoad failed on node {peer_id}")),
        )),
        Ok(Ok(_)) => Err(InferenceOutcome::Error(
            "Unexpected frame type from node during model load".into(),
        )),
        Ok(Err(_)) => Err(InferenceOutcome::Error(
            "Node disconnected during model load".into(),
        )),
        Err(_) => {
            state
                .write()
                .await
                .pending_requests
                .remove(&load_request_id);
            Err(InferenceOutcome::Error(format!(
                "Model load timed out after {MODEL_LOAD_TIMEOUT_SECS}s"
            )))
        }
    }
}

/// Execute one typed inference round on a node.
async fn run_inference(
    run_id: &str,
    round_id: &str,
    round_messages: Vec<ConversationMessage>,
    tool_entries: &[ToolEntry],
    model_id: Option<&str>,
    state: &SharedState,
    session_id: &str,
) -> InferenceOutcome {
    let tools: Vec<_> = tool_entries
        .iter()
        .filter(|tool| tool.available)
        .map(|tool| tool.spec.clone())
        .collect();

    let round_request = RunRoundRequest {
        run_id: run_id.to_string(),
        round_id: round_id.to_string(),
        session_id: session_id.to_string(),
        messages: round_messages,
        tools,
        model_id: model_id.map(str::to_owned),
    };

    let (selected_peer_id, node_sender) = match model_id {
        Some(model_id) => match ensure_model_loaded(model_id, state).await {
            Ok(selection) => selection,
            Err(outcome) => return outcome,
        },
        None => {
            let state_read = state.read().await;
            match state_read.find_any_llm_peer() {
                Some(selection) => selection,
                None => return InferenceOutcome::NoLlmAvailable,
            }
        }
    };

    let forwarded_id = Frame::new_id();
    let forwarded_frame = Frame::Request {
        id: forwarded_id.clone(),
        method: Method::RunRound,
        params: serde_json::to_value(&round_request).unwrap_or_default(),
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
        return InferenceOutcome::Error("Failed to send inference request to node".into());
    }

    match tokio::time::timeout(std::time::Duration::from_secs(120), response_rx).await {
        Ok(Ok(Frame::Response {
            ok: true, payload, ..
        })) => parse_inference_response(payload, selected_peer_id),
        Ok(Ok(Frame::Response {
            ok: false, error, ..
        })) => InferenceOutcome::Error(
            error
                .map(|payload| payload.message)
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

/// Parse a typed round response into an engine outcome.
fn parse_inference_response(
    payload: Option<serde_json::Value>,
    selected_peer_id: String,
) -> InferenceOutcome {
    let Some(payload) = payload else {
        return InferenceOutcome::Error("Empty inference response".into());
    };

    let response: RunRoundResponse = match serde_json::from_value(payload) {
        Ok(response) => response,
        Err(error) => {
            return InferenceOutcome::Error(format!("Invalid round response: {error}"));
        }
    };

    if !response.tool_calls.is_empty() {
        return InferenceOutcome::ToolCalls(ToolCallOutcome {
            calls: response
                .tool_calls
                .into_iter()
                .map(|tool_call| tool_call.call)
                .collect(),
            rationale: response.rationale,
            selected_peer_id,
        });
    }

    let raw_text = response
        .content
        .or(response.rationale.clone())
        .unwrap_or_default();
    if raw_text.trim().is_empty() {
        return InferenceOutcome::Error(
            "Inference returned no assistant content or tool calls".into(),
        );
    }
    InferenceOutcome::Reply(ReplyOutcome {
        raw_text,
        rationale: response.rationale,
        selected_peer_id,
    })
}

/// Strip thinking blocks from model output.
fn strip_thinking_content(raw: &str) -> (String, Option<String>) {
    let mut visible = String::new();
    let mut thinking = String::new();
    let mut rest = raw;

    while let Some(start) = rest.find(THINKING_BLOCK_OPEN) {
        visible.push_str(&rest[..start]);
        let after_open = &rest[start + THINKING_BLOCK_OPEN.len()..];
        if let Some(end) = after_open.find(THINKING_BLOCK_CLOSE) {
            thinking.push_str(after_open[..end].trim());
            rest = &after_open[end + THINKING_BLOCK_CLOSE.len()..];
        } else {
            thinking.push_str(after_open.trim());
            rest = "";
            break;
        }
    }

    visible.push_str(rest);
    let visible = visible.trim().to_string();
    let thinking = if thinking.is_empty() {
        None
    } else {
        Some(thinking)
    };
    (visible, thinking)
}

/// Serialize tool calls into the conversation format used for structured replay.
fn serialize_tool_calls_for_history(calls: &[ToolCall]) -> String {
    serde_json::to_string(calls).unwrap_or_default()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn serialize_tool_calls_for_history_uses_core_tool_calls() {
        let calls = vec![
            ToolCall {
                id: "call-1".into(),
                index: 0,
                name: "notes.list".into(),
                arguments: serde_json::json!({}),
            },
            ToolCall {
                id: "call-2".into(),
                index: 1,
                name: "echo.run".into(),
                arguments: serde_json::json!({
                    "message": "hello",
                    "count": 2,
                    "verbose": true,
                    "tags": ["a", "b"],
                    "meta": {
                        "source": "test"
                    }
                }),
            },
        ];

        let serialized = serialize_tool_calls_for_history(&calls);
        let reparsed: Vec<ToolCall> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(reparsed, calls);
    }

    #[test]
    fn strip_thinking_content_separates_hidden_blocks() {
        let raw = concat!(
            "Visible intro\n",
            "<|channel>thought\nplan the tool call<channel|>",
            "Visible answer"
        );

        let (visible, thinking) = strip_thinking_content(raw);

        assert_eq!(visible, "Visible intro\nVisible answer");
        assert_eq!(thinking.as_deref(), Some("plan the tool call"));
    }

    #[test]
    fn parse_inference_response_prefers_tool_calls() {
        let payload = serde_json::json!({
            "content": "fallback",
            "rationale": "need to call a tool",
            "toolCalls": [
                {
                    "id": "call-1",
                    "index": 0,
                    "name": "notes.list",
                    "arguments": {"limit": 5}
                }
            ]
        });

        let outcome = parse_inference_response(Some(payload), "peer-a".into());

        match outcome {
            InferenceOutcome::ToolCalls(tool_call_outcome) => {
                assert_eq!(tool_call_outcome.selected_peer_id, "peer-a");
                assert_eq!(
                    tool_call_outcome.rationale.as_deref(),
                    Some("need to call a tool")
                );
                assert_eq!(tool_call_outcome.calls.len(), 1);
                assert_eq!(tool_call_outcome.calls[0].name, "notes.list");
            }
            InferenceOutcome::Reply(_)
            | InferenceOutcome::Error(_)
            | InferenceOutcome::NoLlmAvailable => {
                panic!("expected tool call outcome");
            }
        }
    }
}
