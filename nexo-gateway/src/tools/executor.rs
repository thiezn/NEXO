//! Tool execution helpers for the run loop.

use crate::server::state::SharedState;
use nexo_ws_schema::{Frame, Method, ToolsExecuteParams, ToolsExecuteResponse};

/// Execute a tool by preferring gateway-local tools before forwarding to a node.
///
/// # Errors
///
/// Returns an error when the tool is unknown, the hosting node disconnects, the
/// forwarded request times out, or the gateway-local executor fails.
pub async fn execute_tool(
    tool_name: &str,
    args: &serde_json::Value,
    state: &SharedState,
) -> Result<ToolsExecuteResponse, String> {
    let gateway_tool = {
        let state_read = state.read().await;
        state_read.gateway_tools.get_tool(tool_name).cloned()
    };
    if let Some(tool) = gateway_tool {
        return match tool.execute(args.clone()).await {
            Ok(result) => Ok(ToolsExecuteResponse {
                success: result.success,
                output: result.output,
                error: result.error,
            }),
            Err(error) => Err(format!("Gateway tool error: {error}")),
        };
    }

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
        Ok(mut frame) => {
            if let Frame::Request { ref mut id, .. } = frame {
                *id = forwarded_id.clone();
            }
            frame
        }
        Err(error) => return Err(format!("Failed to build tool request: {error}")),
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
            let response: ToolsExecuteResponse = payload
                .and_then(|payload| serde_json::from_value(payload).ok())
                .unwrap_or(ToolsExecuteResponse {
                    success: false,
                    output: String::new(),
                    error: Some("Invalid tool response".into()),
                });
            Ok(response)
        }
        Ok(Ok(Frame::Response { error, .. })) => Ok(ToolsExecuteResponse {
            success: false,
            output: String::new(),
            error: error.map(|payload| payload.message),
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

/// Derive the capability namespace associated with a tool name.
pub fn tool_capability(tool_name: &str) -> String {
    tool_name.split('.').next().unwrap_or(tool_name).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_capability_uses_name_prefix() {
        assert_eq!(tool_capability("echo.run"), "echo");
        assert_eq!(tool_capability("ping"), "ping");
    }
}
