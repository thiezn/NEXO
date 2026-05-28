use crate::transport::send;
use cli_helpers::Error;
use nexo_core::{
    ToolCall, ToolCallId, ToolRegistry, ToolResult, ToolResultContent, ToolResultStatus,
};
use nexo_ws_client::{NexoConnection, WriteHalf};
use nexo_ws_schema::{
    ErrorPayload, Frame, Method, ToolsExecuteParams, ToolsExecuteResponse, ToolsRegisterParams,
};

/// Register all local tools with the gateway after the websocket handshake completes.
pub(crate) async fn register_tools(
    conn: &mut NexoConnection,
    registry: &ToolRegistry,
) -> cli_helpers::Result {
    let specs = registry.definitions();
    let tool_count = specs.len();
    tracing::info!("Registering {tool_count} tool(s) with gateway");

    let register_frame =
        Frame::request(Method::ToolsRegister, &ToolsRegisterParams { tools: specs })
            .map_err(|error| Error::Other(format!("Failed to build register frame: {error}")))?;

    conn.send_frame(&register_frame)
        .await
        .map_err(|error| Error::Other(format!("Failed to send register: {error}")))?;

    await_register_response(conn, tool_count).await
}

async fn await_register_response(
    conn: &mut NexoConnection,
    tool_count: usize,
) -> cli_helpers::Result {
    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|error| Error::Other(format!("Failed to receive register response: {error}")))?
            .ok_or_else(|| Error::Other("Connection closed during registration".into()))?;

        match frame {
            Frame::Response {
                ok: true, payload, ..
            } => {
                let registered = payload
                    .as_ref()
                    .and_then(|value| value.get("registered"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
                tracing::info!("Gateway accepted {registered}/{tool_count} tool(s)");
                return Ok(());
            }
            Frame::Response {
                ok: false, error, ..
            } => {
                let message = error
                    .map(|error| format!("{}: {}", error.code, error.message))
                    .unwrap_or_else(|| "Unknown error".into());
                return Err(Error::Other(format!(
                    "Tool registration rejected: {message}"
                )));
            }
            Frame::Event { .. } => continue,
            other => tracing::warn!("Unexpected frame during registration: {other:?}"),
        }
    }
}

/// Execute a gateway `tools.execute` request against the local tool registry.
pub(crate) async fn handle_tool_execute(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    registry: &ToolRegistry,
) -> cli_helpers::Result {
    let exec_params: ToolsExecuteParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            let error_response = Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "invalid_params".into(),
                    message: format!("Invalid tools.execute params: {error}"),
                },
            );
            send(writer, &error_response).await?;
            return Ok(());
        }
    };

    tracing::info!("Executing tool '{}'", exec_params.tool);
    tracing::debug!("Tool '{}' args: {}", exec_params.tool, exec_params.args);
    let start = std::time::Instant::now();

    let tool_call = ToolCall {
        id: ToolCallId::from(exec_params.idempotency_key.clone()),
        index: 0,
        name: exec_params.tool.clone(),
        arguments: exec_params.args,
    };

    let response = match registry.try_execute(tool_call).await {
        Ok(Some(result)) => {
            let elapsed = start.elapsed();
            let tool_response = response_from_tool_result(result);
            tracing::info!(
                "Tool '{}' completed in {:.2}ms (success={})",
                exec_params.tool,
                elapsed.as_secs_f64() * 1000.0,
                tool_response.success
            );
            tracing::debug!(
                "Tool '{}' output: {}, error: {:?}",
                exec_params.tool,
                tool_response.output,
                tool_response.error
            );
            Frame::ok_response(request_id, &tool_response).unwrap_or_else(|error| {
                Frame::error_response(
                    request_id,
                    ErrorPayload {
                        code: "internal_error".into(),
                        message: error.to_string(),
                    },
                )
            })
        }
        Ok(None) => {
            tracing::warn!("Tool '{}' not found locally", exec_params.tool);
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "tool_not_found".into(),
                    message: format!("Tool '{}' is not available on this node", exec_params.tool),
                },
            )
        }
        Err(error) => Frame::ok_response(
            request_id,
            &ToolsExecuteResponse {
                success: false,
                output: String::new(),
                error: Some(error.to_string()),
            },
        )
        .unwrap_or_else(|error| {
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "internal_error".into(),
                    message: error.to_string(),
                },
            )
        }),
    };

    send(writer, &response).await
}

fn response_from_tool_result(result: ToolResult) -> ToolsExecuteResponse {
    let content = match result.content {
        ToolResultContent::Text(text) => text,
        ToolResultContent::Json(value) => value.to_string(),
    };

    match result.status {
        ToolResultStatus::Success => ToolsExecuteResponse {
            success: true,
            output: content,
            error: None,
        },
        ToolResultStatus::Failure => ToolsExecuteResponse {
            success: false,
            output: String::new(),
            error: Some(content),
        },
    }
}
