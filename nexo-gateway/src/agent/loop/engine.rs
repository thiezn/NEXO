use crate::{agent::persistence, server::state::SharedState, tools};
use nexo_core::{MessageRole, ReasoningSettings};
use nexo_ws_schema::{Frame, RunStatus};
use sqlx::SqlitePool;
use tokio::sync::broadcast;

use super::{
    context_manager::ContextManager,
    events,
    inference::{InferenceOutcome, run_inference},
};

/// Maximum number of rounds before the engine stops a run.
const MAX_ITERATIONS: usize = 20;
pub(super) const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

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
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
    model_id: Option<&str>,
    prompt_collection_id: Option<&str>,
    reasoning: ReasoningSettings,
) {
    if let Err(error) = persistence::insert_conversation_entry(
        db,
        session_id,
        Some(run_id),
        None,
        MessageRole::User,
        input,
        persistence::ConversationEntryKind::UserInput,
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
            MessageRole::System,
            &instructions_string,
            persistence::ConversationEntryKind::Instruction,
            None,
            None,
        )
        .await;
    }

    run_existing(
        run_id,
        session_id,
        db,
        state,
        event_tx,
        model_id,
        prompt_collection_id,
        reasoning,
    )
    .await;
}

/// Drive the round-based loop for an existing run until it completes, fails, or queues.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn run_existing(
    run_id: &str,
    session_id: &str,
    db: &SqlitePool,
    state: &SharedState,
    event_tx: &broadcast::Sender<Frame>,
    model_id: Option<&str>,
    prompt_collection_id: Option<&str>,
    reasoning: ReasoningSettings,
) {
    let _ = crate::agent::locks::reap_expired(db).await;

    let context_manager = ContextManager::new(state, prompt_collection_id).await;

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
                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    persistence::RoundStatus::Failed,
                    None,
                    None,
                )
                .await;
                events::fail_run(event_tx, db, run_id, session_id, &error.to_string()).await;
                return;
            }
        };

        tracing::info!(
            "Run {run_id} entering round {} (round_id={round_id}, messages={}, tools={}, model={:?}, reasoning={:?})",
            round_index,
            prepared_context.persisted_message_count,
            tool_entries.iter().filter(|tool| tool.available).count(),
            model_id,
            reasoning,
        );

        let inference = run_inference(
            run_id,
            &round_id,
            prepared_context.round_messages,
            &tool_entries,
            model_id,
            reasoning.clone(),
            state,
            session_id,
        )
        .await;

        if run_cancelled(db, run_id).await {
            let _ = persistence::finish_round(
                db,
                &round_id,
                persistence::RoundStatus::Cancelled,
                None,
                None,
            )
            .await;
            crate::agent::locks::release_all_for_run(db, run_id)
                .await
                .ok();
            return;
        }

        match inference {
            InferenceOutcome::Reply(reply) => {
                let visible_text = reply.content;
                let thinking_content = reply.rationale.clone();
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
                    MessageRole::Assistant,
                    &visible_text,
                    persistence::ConversationEntryKind::AssistantResponse,
                    None,
                    None,
                )
                .await;
                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    persistence::RoundStatus::Completed,
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
                let assistant_tool_history =
                    serde_json::to_string(&outcome.calls).unwrap_or_default();
                let _ = persistence::insert_conversation_entry(
                    db,
                    session_id,
                    Some(run_id),
                    Some(&round_id),
                    MessageRole::Assistant,
                    &assistant_tool_history,
                    persistence::ConversationEntryKind::ToolCallIntent,
                    None,
                    None,
                )
                .await;

                for call in &outcome.calls {
                    if run_cancelled(db, run_id).await {
                        let _ = persistence::finish_round(
                            db,
                            &round_id,
                            persistence::RoundStatus::Cancelled,
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
                                    persistence::ToolTraceStatus::Failed,
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
                                MessageRole::Tool,
                                &output,
                                persistence::ConversationEntryKind::ToolResult,
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
                                    persistence::ToolTraceStatus::Failed,
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
                            persistence::ToolTraceStatus::from_success(success),
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
                        MessageRole::Tool,
                        &output,
                        persistence::ConversationEntryKind::ToolResult,
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
                    persistence::RoundStatus::Completed,
                    outcome.rationale.as_deref(),
                    Some(&outcome.selected_peer_id),
                )
                .await;
                continue;
            }
            InferenceOutcome::Error(error) => {
                tracing::info!("Run {run_id} round {round_id} failed: {error}");
                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    persistence::RoundStatus::Failed,
                    None,
                    None,
                )
                .await;
                events::fail_run(event_tx, db, run_id, session_id, &error).await;
                return;
            }
            InferenceOutcome::NoLlmAvailable => {
                tracing::info!(
                    "Run {run_id} round {round_id} queued (no inference node available)"
                );
                let _ = persistence::finish_round(
                    db,
                    &round_id,
                    persistence::RoundStatus::Queued,
                    None,
                    None,
                )
                .await;
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
