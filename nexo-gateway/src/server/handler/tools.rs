//! WebSocket handlers for tool registration and execution.

use crate::server::state::SharedState;
use nexo_ws_schema::{
    ErrorPayload, Frame, Method, Role, ToolsRegisterParams, ToolsRegisterResponse,
};

use super::base::{
    ForwardErrorCodes, TOOL_EXECUTION_TIMEOUT, forward_to_node, ok_or_internal_error, parse_params,
};

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
            Some(peer) if peer.role == Role::Node => {}
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
    let tool_name = match params.get("tool").and_then(|v| v.as_str()) {
        Some(name) => name.to_owned(),
        None => {
            return Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "invalid_params".into(),
                    message: "Missing 'tool' field in tools.execute params".into(),
                },
            );
        }
    };

    tracing::info!("Routing tools.execute for '{tool_name}' (requested by peer {peer_id})");

    let node_sender = {
        let state_read = state.read().await;
        let tool = match state_read.find_tool(&tool_name) {
            Some(t) => t,
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "tool_not_found".into(),
                        message: format!("Tool '{tool_name}' is not registered"),
                    },
                );
            }
        };
        match state_read.peer_senders.get(&tool.peer_id) {
            Some(s) => s.clone(),
            None => {
                return Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "tool_unavailable".into(),
                        message: format!("Node hosting tool '{tool_name}' is not connected"),
                    },
                );
            }
        }
    };

    forward_to_node(
        request_id,
        Method::ToolsExecute,
        params,
        node_sender,
        state,
        TOOL_EXECUTION_TIMEOUT,
        ForwardErrorCodes {
            error_code: "tool_unavailable",
            label: "Tool execution",
        },
    )
    .await
}
