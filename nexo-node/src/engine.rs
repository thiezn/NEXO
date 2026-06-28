use crate::NexoNodeConfig;
use crate::{Error, Result};
use futures_util::StreamExt;
use nexo_ai::InferenceEngine;
use nexo_core::{ClientKind, InferenceRequest, ModelId, OperationId, ToolCall, ToolRegistry};
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::GatewayToNodeMessage;
use nexo_ws_schema::{
    ExecuteToolEvent, Frame, InferenceEvent, LoadModelEvent, NexoEvent, NexoResponse,
    NodeToGatewayMessage, UnloadModelEvent,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tracing::{debug, error, info, warn};

/// Central coordinator for nexo-node, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub(crate) struct NexoNode {
    /// The configuration for the node, including gateway URL, auth token, and node identity.
    config: NexoNodeConfig,

    /// The registry of tools available on this node.
    registry: Arc<ToolRegistry>,

    /// The inference engine responsible for running models and generating responses.
    engine: Arc<InferenceEngine>,
}

impl NexoNode {
    /// Initializes a new NexoNode with the given configuration and tool registry.
    pub fn new(config: NexoNodeConfig, registry: ToolRegistry) -> Result<Self> {
        let catalog = nexo_ai::ModelCatalog::new();
        let local_available_manifests = catalog.list_downloaded_manifests();

        let engine = Arc::new(InferenceEngine::new(local_available_manifests)?);
        let registry = Arc::new(registry);

        Ok(Self {
            config,
            registry,
            engine,
        })
    }

    /// Connect to the nexo-gateway.
    async fn connect(&self) -> Result<NexoConnection> {
        let url = &self.config.gateway_url;
        info!(url = url, "Connecting to gateway...");

        // TODO: Probably we should remove NexoNodeConfig and only have ClientKind, OR better, NodeProperties stored in self.config?
        let client_kind = ClientKind::new_node(
            &self.config.node_id,
            &self.config.node_version,
            self.config.platform,
            &self.config.device_id,
            self.registry.capability_names(),
            self.registry.tool_names(),
            self.engine
                .model_ids()
                .into_iter()
                .map(|id| id.clone())
                .collect(),
        );

        let mut conn = NexoConnection::connect(url, &self.config.auth_token, client_kind).await?;

        // TODO: Do we actually need to run the tools register? Why not roll this all up
        // in the connect handshake? Also, the capabilities names and command_names
        // are probably not the right abstraction. I want to be able to pass ToolDefinition
        // in my connect call, and the gateway, user and nodes should understand
        // what commands and capabilities map to what tools.
        //
        // The register_tools function could still be useful if we expect to be able to
        // update existing tools without reconnecting?
        if self.registry.len() > 0 {
            info!(
                tool_count = self.registry.len(),
                "Registering tools with gateway..."
            );
            register_tools(&mut conn, &self.registry).await?;
        } else {
            info!("No tools to register with gateway");
        }

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
                                // break; // Exit loop on receive error to trigger reconnect. Maybe we should raise an error
                            }
                        }
                    }

                    // Handle results of actions taken in response to gateway messages
                    Some(msg) = rx.recv() => {
                        if msg == NodeToGatewayMessage::Disconnect {
                            info!("Disconnecting from gateway...");
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
                response.result();
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
                    load_model(operation_id, &engine, model_id, tx).await;
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
            GatewayToNodeMessage::Cancel(request) => {
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
            GatewayToNodeMessage::RegisterTools(_) | GatewayToNodeMessage::Connect(_) => {
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

/// Register all local tools with the gateway after the websocket handshake completes.
///
/// # Arguments
///
/// * `conn` - The active websocket connection to the gateway.
/// * `registry` - The local tool registry containing the tools to register.
async fn register_tools(conn: &mut NexoConnection, registry: &ToolRegistry) -> Result {
    let tools = registry.definitions();

    let register_frame = Frame::new(NodeToGatewayMessage::RegisterTools(tools.clone()))?;

    conn.send_frame(&register_frame).await?;

    loop {
        let frame = conn.recv_frame().await?;

        let (frame_id, payload) = frame.into_parts::<GatewayToNodeMessage>()?;
        info!(frame_id = ?frame_id, "Received frame");

        match payload {
            GatewayToNodeMessage::RegisterTools(response) => match response {
                NexoResponse::Accepted { operation_id } => {
                    error!(
                        operation_id = ?operation_id,
                        "Tool registration generated accepted response, expecting a synchronous completed response"
                    );
                    return Err(Error::ToolRegistration {
                        operation_id,
                        error: "Tool registration generated accepted response, expecting a synchronous completed response".into(),
                    });
                }
                NexoResponse::Completed { operation_id, .. } => {
                    info!(operation_id = ?operation_id, "Tool registration completed");
                    return Ok(());
                }
                NexoResponse::Failed {
                    operation_id,
                    error,
                } => {
                    error!(operation_id = ?operation_id, error = ?error, "Tool registration failed");
                    return Err(Error::ToolRegistration {
                        operation_id,
                        error: error.to_string(),
                    });
                }
            },
            other => {
                warn!(message = ?other, "Unexpected frame during registration")
            }
        }
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
    tx.send(NodeToGatewayMessage::StartInferenceRunEvent(
        NexoEvent::Correlated {
            operation_id: operation_id.clone(),
            event: InferenceEvent::RunStarted {
                operation_id: operation_id.clone(),
                run_id: request.run_id.clone(),
            },
        },
    ))
    .await?;

    let mut stream = engine.run_inference(request.clone()).await?;

    while let Some(item) = stream.next().await {
        match item {
            Ok(_) => {
                todo!("Implement proper chunking etc")
                // tx.send(NodeToGatewayMessage::StartInferenceRunEvent(
                //     NexoEvent::Correlated {
                //         operation_id,
                //         event: InferenceEvent::Chunk {
                //             seq: 1,
                //             output: response,
                //         },
                //     },
                // ))
                // .await?;
            }
            Err(err) => {
                error!(operation_id = %operation_id, error = ?err, "Inference run failed");
                tx.send(NodeToGatewayMessage::StartInferenceRunEvent(
                    NexoEvent::Correlated {
                        operation_id,
                        event: InferenceEvent::Failed {
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
