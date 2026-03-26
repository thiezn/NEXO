use crate::config::NodeConfig;
use crate::registry::ToolRegistry;
use nexo_ws_client::{NexoConnection, default_node_connect_params, perform_handshake};
use nexo_ws_schema::{
    ErrorPayload, Frame, Method, ToolsExecuteParams, ToolsExecuteResponse, ToolsRegisterParams,
};
use std::time::Duration;

/// Run the node, connecting to the gateway and reconnecting on disconnect.
pub async fn run_node(config: &NodeConfig, registry: &ToolRegistry) -> utl_helpers::Result {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        tracing::info!(
            "Connecting to gateway at {} (attempt {attempt})",
            config.gateway_url
        );

        match connect_and_run(config, registry).await {
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
) -> utl_helpers::Result {
    // Step 1: Connect to gateway
    let mut conn = NexoConnection::connect(&config.gateway_url, &config.auth_token)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Connection failed: {e}")))?;

    tracing::info!("Connected to gateway");

    // Step 2: Handshake
    let (capabilities, commands) = registry.capabilities_and_commands();
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

    // Wait for register response
    let register_response = conn
        .recv_frame()
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Failed to receive register response: {e}")))?
        .ok_or_else(|| utl_helpers::Error::Network("Connection closed during registration".into()))?;

    match &register_response {
        Frame::Response {
            ok: true, payload, ..
        } => {
            let registered = payload
                .as_ref()
                .and_then(|p| p.get("registered"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            tracing::info!("Gateway accepted {registered}/{tool_count} tool(s)");
        }
        Frame::Response {
            ok: false, error, ..
        } => {
            let msg = error
                .as_ref()
                .map(|e| format!("{}: {}", e.code, e.message))
                .unwrap_or_else(|| "Unknown error".into());
            return Err(utl_helpers::Error::Network(format!(
                "Tool registration rejected: {msg}"
            )));
        }
        _ => {
            tracing::warn!("Unexpected frame during registration: {register_response:?}");
        }
    }

    tracing::info!("Node ready, listening for tool execution requests");

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
