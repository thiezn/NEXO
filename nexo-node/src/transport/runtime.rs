use super::inference::{
    InferenceResult, InferenceSender, SharedModels, handle_model_load, handle_model_unload,
    load_startup_models, push_model_status, queue_image_analyze, queue_run_round, shared_models,
};
use super::protocol::{send, send_busy_error};
use crate::config::NodeConfig;
use crate::tools::{handle_tool_execute, register_tools};
use cli_helpers::Error;
use nexo_ai::RegisteredModelConfig;
use nexo_core::ToolRegistry;
use nexo_ws_client::{
    NexoConnection, ReadHalf, WriteHalf, default_node_connect_params, perform_handshake,
};
use nexo_ws_schema::{ErrorPayload, EventKind, Frame, Method};
use std::time::Duration;

enum MessageLoopAction {
    Continue,
    Stop,
}

struct GatewayFrameContext<'a> {
    registry: &'a ToolRegistry,
    models: &'a SharedModels,
    inference_tx: &'a InferenceSender,
    inference_busy: &'a mut bool,
}

/// Run the node, connecting to the gateway and reconnecting on disconnect.
///
/// # Arguments
///
/// * `config` - Node-level gateway and runtime configuration.
/// * `registry` - Local tool registry exposed by this node.
/// * `available_models` - Downloaded models that `nexo-ai` can load for this node.
///
/// # Errors
///
/// Returns an error if the node cannot maintain a healthy connection to the gateway.
pub async fn run_node(
    config: &NodeConfig,
    registry: &ToolRegistry,
    available_models: Vec<RegisteredModelConfig>,
) -> cli_helpers::Result {
    let models = shared_models(config.runtime.clone(), available_models);
    load_startup_models(&models, config).await;

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        tracing::info!(
            "Connecting to gateway at {} (attempt {attempt})",
            config.gateway_url
        );

        match connect_and_run(config, registry, models.clone()).await {
            Ok(()) => {
                tracing::info!("Node disconnected gracefully");
                break;
            }
            Err(error) => {
                tracing::warn!(
                    "Connection lost: {error}. Reconnecting in {}ms...",
                    config.reconnect_interval_ms
                );
                tokio::time::sleep(Duration::from_millis(config.reconnect_interval_ms)).await;
            }
        }
    }
    Ok(())
}

/// Connect to the gateway, initialize the node session, and process frames until disconnect.
async fn connect_gateway(
    config: &NodeConfig,
    registry: &ToolRegistry,
    models: &SharedModels,
) -> cli_helpers::Result<NexoConnection> {
    let mut conn = NexoConnection::connect(&config.gateway_url, &config.auth_token)
        .await
        .map_err(|error| Error::Other(format!("Connection failed: {error}")))?;

    tracing::info!("Connected to gateway");

    let (capabilities, commands) = registry.capabilities_and_commands();
    tracing::debug!("Handshaking with capabilities={capabilities:?}, commands={commands:?}");
    let available_models = {
        let models = models.lock().await;
        models.available_model_ids()
    };

    let params = default_node_connect_params(
        &config.node_id,
        &config.node_version,
        config.platform,
        &config.device_id,
        capabilities,
        commands,
        available_models,
    );

    let hello = perform_handshake(&mut conn, params)
        .await
        .map_err(|error| Error::Other(format!("Handshake failed: {error}")))?;

    tracing::info!(
        "Handshake complete: protocol v{}, tick interval {}ms",
        hello.protocol,
        hello.policy.tick_interval_ms
    );

    Ok(conn)
}

async fn connect_and_run(
    config: &NodeConfig,
    registry: &ToolRegistry,
    models: SharedModels,
) -> cli_helpers::Result {
    let mut conn = connect_gateway(config, registry, &models).await?;
    register_tools(&mut conn, registry).await?;

    tracing::info!("Node ready, listening for requests");

    let (mut writer, mut reader) = conn.into_split();
    push_model_status(&mut writer, &models).await;

    run_message_loop(&mut writer, &mut reader, registry, &models).await?;

    if let Err(error) = writer.close().await {
        tracing::debug!("Close error (non-fatal): {error}");
    }

    Ok(())
}

async fn run_message_loop(
    writer: &mut WriteHalf,
    reader: &mut ReadHalf,
    registry: &ToolRegistry,
    models: &SharedModels,
) -> cli_helpers::Result {
    let (inference_tx, mut inference_rx) =
        tokio::sync::mpsc::channel::<(String, InferenceResult)>(1);
    let mut inference_busy = false;

    loop {
        tokio::select! {
            frame = reader.recv_frame() => {
                let frame = frame
                    .map_err(|error| Error::Other(format!("Receive error: {error}")))?;
                let context = GatewayFrameContext {
                    registry,
                    models,
                    inference_tx: &inference_tx,
                    inference_busy: &mut inference_busy,
                };
                let action = handle_gateway_frame(writer, frame, context).await?;

                if matches!(action, MessageLoopAction::Stop) {
                    break;
                }
            }

            Some((request_id, result)) = inference_rx.recv() => {
                inference_busy = false;
                handle_inference_result(writer, &request_id, result).await?;
            }
        }
    }

    Ok(())
}

async fn handle_gateway_frame(
    writer: &mut WriteHalf,
    frame: Option<Frame>,
    context: GatewayFrameContext<'_>,
) -> cli_helpers::Result<MessageLoopAction> {
    match frame {
        Some(Frame::Request {
            id,
            method: Method::ToolsExecute,
            params,
        }) => {
            handle_tool_execute(writer, &id, params, context.registry).await?;
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Request {
            id,
            method: Method::RunRound,
            params,
        }) => {
            if *context.inference_busy {
                send_busy_error(writer, &id).await?;
            } else {
                *context.inference_busy = true;
                queue_run_round(&id, params, context.models, context.inference_tx).await?;
            }
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Request {
            id,
            method: Method::ImageAnalyze,
            params,
        }) => {
            if *context.inference_busy {
                send_busy_error(writer, &id).await?;
            } else {
                *context.inference_busy = true;
                queue_image_analyze(&id, params, context.models, context.inference_tx).await?;
            }
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Request {
            id,
            method: Method::ModelLoad,
            params,
        }) => {
            handle_model_load(writer, &id, params, context.models).await?;
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Request {
            id,
            method: Method::ModelUnload,
            params,
        }) => {
            handle_model_unload(writer, &id, params, context.models).await?;
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Event {
            event: EventKind::Tick,
            ..
        }) => {
            tracing::trace!("Received tick");
            Ok(MessageLoopAction::Continue)
        }
        Some(Frame::Event {
            event: EventKind::Shutdown,
            payload,
            ..
        }) => {
            let reason = payload
                .get("reason")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            tracing::info!("Received shutdown event: {reason}");
            Ok(MessageLoopAction::Stop)
        }
        Some(Frame::Event { event, .. }) => {
            tracing::debug!("Received event: {event:?}");
            Ok(MessageLoopAction::Continue)
        }
        Some(frame) => {
            tracing::debug!("Received unexpected frame: {frame:?}");
            Ok(MessageLoopAction::Continue)
        }
        None => Err(Error::Other("Connection closed by gateway".into())),
    }
}

async fn handle_inference_result(
    writer: &mut WriteHalf,
    request_id: &str,
    result: InferenceResult,
) -> cli_helpers::Result {
    let response = match result {
        Ok(payload) => Frame::ok_response(request_id, &payload).unwrap_or_else(|error| {
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "internal_error".into(),
                    message: error.to_string(),
                },
            )
        }),
        Err(message) => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "inference_error".into(),
                message,
            },
        ),
    };

    send(writer, &response).await
}
