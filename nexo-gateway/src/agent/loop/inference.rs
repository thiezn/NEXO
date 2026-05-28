use crate::server::state::SharedState;
use nexo_core::{ConversationMessage, ReasoningSettings, ToolCall};
use nexo_ws_schema::{Frame, Method, RunRoundRequest, RunRoundResponse, ToolEntry};

use super::router::Router;

/// Normalized result of one gateway-to-node inference round.
pub(super) enum InferenceOutcome {
    Reply(ReplyOutcome),
    ToolCalls(ToolCallOutcome),
    Error(String),
    NoLlmAvailable,
}

/// Terminal assistant reply for a round.
pub(super) struct ReplyOutcome {
    pub(super) content: String,
    pub(super) rationale: Option<String>,
    pub(super) selected_peer_id: String,
}

/// Tool-call plan returned for a round.
pub(super) struct ToolCallOutcome {
    pub(super) calls: Vec<ToolCall>,
    pub(super) rationale: Option<String>,
    pub(super) selected_peer_id: String,
}

/// Execute one inference round on a node.
#[expect(clippy::too_many_arguments)]
pub(super) async fn run_inference(
    run_id: &str,
    round_id: &str,
    round_messages: Vec<ConversationMessage>,
    tool_entries: &[ToolEntry],
    model_id: Option<&str>,
    reasoning: ReasoningSettings,
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
        reasoning,
        model_id: model_id.map(str::to_owned),
    };

    let (selected_peer_id, node_sender) = match Router::route_inference(state, model_id).await {
        Ok(selection) => selection,
        Err(outcome) => return outcome,
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

    let content = response.content.unwrap_or_default();
    if content.trim().is_empty() {
        return InferenceOutcome::Error(
            "Inference returned no assistant content or tool calls".into(),
        );
    }

    InferenceOutcome::Reply(ReplyOutcome {
        content,
        rationale: response.rationale,
        selected_peer_id,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

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

    #[test]
    fn parse_inference_response_keeps_rationale_out_of_visible_content() {
        let payload = serde_json::json!({
            "content": "Visible answer",
            "rationale": "hidden reasoning"
        });

        let outcome = parse_inference_response(Some(payload), "peer-a".into());

        match outcome {
            InferenceOutcome::Reply(reply) => {
                assert_eq!(reply.content, "Visible answer");
                assert_eq!(reply.rationale.as_deref(), Some("hidden reasoning"));
            }
            InferenceOutcome::ToolCalls(_)
            | InferenceOutcome::Error(_)
            | InferenceOutcome::NoLlmAvailable => {
                panic!("expected reply outcome");
            }
        }
    }
}
