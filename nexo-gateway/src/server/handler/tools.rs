//! WebSocket handlers for tool registration and execution.

use crate::server::state::SharedState;
use nexo_core::{ToolCall, ToolCallId};
use nexo_ws_schema::{
    ConnectionRole, ErrorPayload, Frame, ToolsExecuteParams, ToolsRegisterParams,
    ToolsRegisterResponse,
};

use super::base::{ok_or_internal_error, parse_params};

/// Handle `tools.register` requests from node peers.
pub(super) async fn handle_register(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
) -> Frame {
    {
        let state_read = state.read().await;
        match state_read.peers.get(peer_id) {
            Some(peer) if peer.role == ConnectionRole::Node => {}
            Some(_) => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "forbidden".into(),
                        message: "Only nodes can register tools".into(),
                    },
                );
            }
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "unknown_peer".into(),
                        message: "Peer not found in state".into(),
                    },
                );
            }
        }
    }

    let register_params: ToolsRegisterParams =
        match parse_params(request_id, params, "tools.register") {
            Ok(p) => p,
            Err(f) => return f,
        };

    let tool_count = register_params.tools.len();
    let tool_names: Vec<String> = register_params
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let registered = {
        let mut state_write = state.write().await;
        state_write.register_tools(peer_id, register_params.tools)
    };

    tracing::info!("Node {peer_id} registered {registered}/{tool_count} tool(s): {tool_names:?}",);

    ok_or_internal_error(request_id, ToolsRegisterResponse { registered })
}

/// Handle `tools.execute` requests by forwarding them to the hosting node.
pub(super) async fn handle_execute(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
) -> Frame {
    let exec_params: ToolsExecuteParams = match parse_params(request_id, params, "tools.execute") {
        Ok(params) => params,
        Err(frame) => return frame,
    };

    tracing::info!(
        "Routing tools.execute for '{}' (requested by peer {peer_id})",
        exec_params.tool
    );

    let call = ToolCall {
        id: ToolCallId::from(exec_params.idempotency_key),
        index: 0,
        name: exec_params.tool,
        arguments: exec_params.args,
    };

    match crate::tools::execute_tool(call, state).await {
        Ok(response) => ok_or_internal_error(request_id, response),
        Err(message) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: tool_error_code(&message).into(),
                message,
            },
        ),
    }
}

fn tool_error_code(message: &str) -> &'static str {
    if message.starts_with("Tool '") && message.ends_with("' not found") {
        "tool_not_found"
    } else {
        "tool_unavailable"
    }
}
