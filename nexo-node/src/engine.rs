use crate::Result;
use futures_util::StreamExt;
use nexo_ai::InferenceEngine;
use nexo_core::{
    ClientKind, InferenceRequest, ModelId, NodeProperties, OperationId, ToolCall, ToolRegistry,
};
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::{
    ExecuteToolEvent, Frame, GatewayToNodeMessage, InferenceRunEvent, LoadModelEvent, NexoEvent,
    NexoResponse, NodeToGatewayMessage, UnloadModelEvent,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tracing::{debug, error, info, warn};
/// Central coordinator for nexo-node, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub struct NexoNode {
    /// The configuration for the node, including gateway URL, auth token, and node identity.
    config: NodeProperties,

    /// The registry of tools available on this node.
    registry: Arc<ToolRegistry>,

    /// The inference engine responsible for running models and generating responses.
    engine: Arc<InferenceEngine>,
}

impl NexoNode {
    /// Initializes a new NexoNode from prepared properties and runtime dependencies.
    pub fn new(
        config: NodeProperties,
        registry: Arc<ToolRegistry>,
        engine: Arc<InferenceEngine>,
    ) -> Self {
        Self {
            config,
            registry,
            engine,
        }
    }

    /// Connect to the nexo-gateway.
    async fn connect(&self) -> Result<NexoConnection> {
        let url = self.config.gateway_url();
        info!(url = url, "Connecting to gateway...");

        let conn = NexoConnection::connect(url, ClientKind::Node(self.config.clone())).await?;

        info!("Node setup complete, entering main loop");
        Ok(conn)
    }

    /// Start the NexoNode runtime, connect to the gateway, and begin processing messages.
    pub async fn run(&self) -> Result {
        let mut conn = self.connect().await?;

        let (tx, mut rx) = mpsc::channel::<NodeToGatewayMessage>(100);

        loop {
            tokio::select! {

                    // Handle incoming frames from the gateway
                    frame = conn.recv_frame() => {
                        match frame {
                            Ok(frame) => {
                                self.handle_frame(frame, tx.clone()).await?;
                            }
                            Err(e) => {
                                error!("Websocket receive error: {e}");
                                return Err(e.into());
                            }
                        }
                    }

                    // Handle results of actions taken in response to gateway messages
                    Some(msg) = rx.recv() => {

                        if msg == NodeToGatewayMessage::Disconnect {
                            info!("Disconnecting from gateway...");
                            let frame = Frame::new(msg)?;
                            conn.send_frame(&frame).await?;
                            break;
                        }

                        let frame = Frame::new(msg)?;
                        conn.send_frame(&frame).await?;
                    }
            }
        }

        if let Err(error) = conn.close().await {
            debug!("Close error (non-fatal): {error}");
        }

        Ok(())
    }

    /// Handle an incoming Frame from the gateway.
    ///
    /// The loop does not process the next frame until this function returns,
    /// so any long-running operations should be offloaded to a separate task.
    ///
    /// The obvious candidates for offloading are inference and tool call operations, which
    /// can take a long time to complete.
    async fn handle_frame(&self, frame: Frame, tx: Sender<NodeToGatewayMessage>) -> Result {
        let (frame_id, payload) = frame.into_parts::<GatewayToNodeMessage>()?;
        info!(frame_id = ?frame_id, "Received frame");

        // Guidelines for handling incoming messages:
        //
        // - Messages that are informational only, we can leverage the response.result() helper to log the outcome and return early.
        // - Messages that do not need a reply and can be handled 'immediately' can be handled inline.
        // - Messages that require a response and are long-running should be offloaded to a separate task. Make sure to first send
        //   an Accepted response back to the gateway before offloading the work to a task. This will ensure the gateway knows the request is
        //   being processed and knows to expect follow up events.
        match payload {
            GatewayToNodeMessage::Disconnect(response) => {
                let _ = response.result();
            }
            GatewayToNodeMessage::LoadModel {
                operation_id,
                model_id,
            } => {
                let engine = self.engine.clone();
                let tx = tx.clone();

                tx.send(NodeToGatewayMessage::LoadModel(NexoResponse::Accepted {
                    operation_id,
                }))
                .await?;

                tokio::spawn(async move {
                    let _ = load_model(operation_id, &engine, model_id, tx).await;
                });
            }
            GatewayToNodeMessage::UnloadModel {
                operation_id,
                model_id,
            } => {
                let engine = self.engine.clone();
                let tx = tx.clone();

                tx.send(NodeToGatewayMessage::UnloadModel(NexoResponse::Accepted {
                    operation_id,
                }))
                .await?;

                tokio::spawn(
                    async move { unload_model(operation_id, &engine, model_id, tx).await },
                );
            }
            GatewayToNodeMessage::StartInferenceRun {
                operation_id,
                request,
            } => {
                let engine = self.engine.clone();
                let tx = tx.clone();

                tx.send(NodeToGatewayMessage::StartInferenceRun(
                    NexoResponse::Accepted { operation_id },
                ))
                .await?;

                tokio::spawn(async move {
                    start_inference_run(operation_id, &engine, request, tx).await
                });
            }
            GatewayToNodeMessage::Cancel(_) => {
                todo!(
                    "Cancel request handling not implemented yet. I think I want to remove the generic cancel request in favor of specific ones. This will ensure the request can contain all required information to make the call instead of maintaining that state in the nexo node memory."
                );
            }
            GatewayToNodeMessage::ExecuteTool {
                operation_id,
                tool_call,
            } => {
                let registry = self.registry.clone();
                let tx = tx.clone();

                tx.send(NodeToGatewayMessage::ExecuteTool(NexoResponse::Accepted {
                    operation_id,
                }))
                .await?;

                tokio::spawn(
                    async move { execute_tool(operation_id, &registry, tool_call, tx).await },
                );
            }
            GatewayToNodeMessage::Connect(_) => {
                let name: &'static str = (&payload).into();
                warn!(
                    name = name,
                    "Received unexpected message from gateway, ignoring"
                );
            }
        };

        Ok(())
    }
}

/// Load a model using the inference engine and generate the appropriate NodeToGatewayMessages
/// to send back to the gateway.
async fn load_model(
    operation_id: OperationId,
    engine: &InferenceEngine,
    model_id: ModelId,
    tx: mpsc::Sender<NodeToGatewayMessage>,
) -> Result {
    // Inform the gateway that the load model request has started processing.
    let started = NodeToGatewayMessage::LoadModelEvent(NexoEvent::Correlated {
        operation_id: operation_id.clone(),
        event: LoadModelEvent::Started {
            model_id: model_id.clone(),
        },
    });
    tx.send(started).await?;

    // Load the model using the inference engine and send the appropriate event back to the gateway.
    match engine.load_model(&model_id).await {
        Ok(_) => {
            info!(model_id = %model_id, "Model loaded successfully");
            tx.send(NodeToGatewayMessage::LoadModelEvent(
                NexoEvent::Correlated {
                    operation_id: operation_id.clone(),
                    event: LoadModelEvent::Completed { model_id },
                },
            ))
            .await?;
        }
        Err(e) => {
            error!(model_id = %model_id, error = ?e, "Failed to load model");
            tx.send(NodeToGatewayMessage::LoadModelEvent(
                NexoEvent::Correlated {
                    operation_id: operation_id,
                    event: LoadModelEvent::Failed {
                        model_id,
                        error: e.to_string(),
                    },
                },
            ))
            .await?;
        }
    }

    Ok(())
}

/// Unload a model using the inference engine and generate the appropriate NodeToGatewayMessages
/// to send back to the gateway.
///
/// Arguments
///
/// * `operation_id` - The unique identifier for the unload operation.
/// * `engine` - The inference engine responsible for managing models.
/// * `model_id` - The identifier of the model to be unloaded.
/// * `tx` - The channel sender to send messages back to the gateway.
async fn unload_model(
    operation_id: OperationId,
    engine: &InferenceEngine,
    model_id: ModelId,
    tx: mpsc::Sender<NodeToGatewayMessage>,
) -> Result {
    // Inform the gateway that the unload model request has started processing.
    let started = NodeToGatewayMessage::UnloadModelEvent(NexoEvent::Correlated {
        operation_id: operation_id.clone(),
        event: UnloadModelEvent::Started {
            model_id: model_id.clone(),
        },
    });
    tx.send(started).await?;

    // Unload the model using the inference engine and send the appropriate event back to the gateway.
    match engine.unload_model(&model_id).await {
        Ok(_) => {
            info!(model_id = %model_id, "Model unloaded successfully");
            tx.send(NodeToGatewayMessage::UnloadModelEvent(
                NexoEvent::Correlated {
                    operation_id: operation_id.clone(),
                    event: UnloadModelEvent::Completed { model_id },
                },
            ))
            .await?;
        }
        Err(e) => {
            error!(model_id = %model_id, error = ?e, "Failed to unload model");
            tx.send(NodeToGatewayMessage::UnloadModelEvent(
                NexoEvent::Correlated {
                    operation_id: operation_id,
                    event: UnloadModelEvent::Failed {
                        model_id,
                        error: e.to_string(),
                    },
                },
            ))
            .await?;
        }
    }

    Ok(())
}

/// Start an inference run using the inference engine and generate the appropriate NodeToGatewayMessages
/// to send back to the gateway.
///
/// # Arguments
///
/// * `operation_id` - The unique identifier for the inference operation.
/// * `engine` - The inference engine responsible for running the model.
/// * `request` - The inference request containing the model selection and operation details.
/// * `tx` - The channel sender to send messages back to the gateway.
async fn start_inference_run(
    operation_id: OperationId,
    engine: &InferenceEngine,
    request: InferenceRequest,
    tx: mpsc::Sender<NodeToGatewayMessage>,
) -> Result {
    let model_id = request.model(engine.model_definitions())?.clone();
    let failure_meta = nexo_core::InferenceMeta::from_request_and_model(&request, model_id);
    let mut stream = engine.run_inference(request).await?;

    while let Some(item) = stream.next().await {
        match item {
            Ok(update) => {
                tx.send(NodeToGatewayMessage::StartInferenceRunEvent(
                    NexoEvent::Correlated {
                        operation_id: operation_id.clone(),
                        event: update.into(),
                    },
                ))
                .await?;
            }
            Err(err) => {
                error!(operation_id = %operation_id, error = ?err, "Inference run failed");
                tx.send(NodeToGatewayMessage::StartInferenceRunEvent(
                    NexoEvent::Correlated {
                        operation_id,
                        event: InferenceRunEvent::Failed {
                            meta: failure_meta,
                            error: err.to_string(),
                        },
                    },
                ))
                .await?;
                return Err(err.into());
            }
        }
    }

    Ok(())
}

/// Execute a tool using the inference engine and generate the appropriate NodeToGatewayMessages
/// to send back to the gateway.
///
/// # Arguments
///
/// * `operation_id` - The unique identifier for the tool execution operation.
/// * `registry` - The tool registry containing the available tools.
/// * `tool_call` - The tool execution request containing the tool name and parameters.
async fn execute_tool(
    operation_id: OperationId,
    registry: &ToolRegistry,
    tool_call: ToolCall,
    tx: mpsc::Sender<NodeToGatewayMessage>,
) -> Result {
    tx.send(NodeToGatewayMessage::ExecuteToolEvent(
        NexoEvent::Correlated {
            operation_id: operation_id.clone(),
            event: ExecuteToolEvent::Started {
                operation_id: operation_id.clone(),
                tool_call_id: tool_call.id.clone(),
            },
        },
    ))
    .await?;

    info!(name = %tool_call.name.clone(), "Executing tool");
    let start = std::time::Instant::now();

    match registry.try_execute(tool_call.clone()).await {
        Ok(result) => {
            let elapsed = start.elapsed();

            info!(name = %tool_call.name.clone(), status = ?result.status, "Tool completed in {:.2}ms",  elapsed.as_secs_f64() * 1000.0);
            debug!(
                name = %tool_call.name.clone(), content= ?result.content, "Tool content",
            );
        }
        Err(err) => {
            error!(name = %tool_call.name.clone(), error = ?err, "Error executing tool");
            tx.send(NodeToGatewayMessage::ExecuteToolEvent(
                NexoEvent::Correlated {
                    operation_id: operation_id.clone(),
                    event: ExecuteToolEvent::Failed {
                        operation_id: operation_id.clone(),
                        tool_call_id: tool_call.id.clone(),
                        error: format!("Tool '{}' is not available on this node", tool_call.name),
                    },
                },
            ))
            .await?;
        }
    };

    Ok(())
}
