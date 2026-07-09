use crate::Result;
use crate::memory::db::DbClient;
use nexo_core::{
    CompactionRequest, ConversationMessage, InferenceIntent, InferenceRequest, ModelId, NexoState,
    Node, OperationId, PeerId, ToolCall, User,
};
use nexo_ws_schema::{InferenceRunEvent, NexoEvent};
use std::collections::VecDeque;
use strum::IntoStaticStr;
use tokio::sync::mpsc;
use tokio::time::{self, Duration};
use tracing::{info, warn};

const QUEUE_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// A single job that the NexoAgent can perform.
///
/// These jobs are queued up in the NexoAgent queue. The agent is responsible
/// for handling parallelism/sequencing.
#[derive(Debug, IntoStaticStr, PartialEq)]
enum AgentJob {
    /// A job to run an inference request.
    RunInference {
        operation_id: OperationId,
        intent: InferenceIntent,
    },
    // TODO: A job to run a tool call, useful for cron jobs.
    // RunTool(ToolCall),
}

/// Tracking the state of a single Inference run.
enum InferenceRunState {
    /// Transforming InferenceIntent into a full InferenceRequest, including context, system prompt, etc.
    PreparingContext {
        user_peer_id: PeerId,
        operation_id: OperationId,
    },

    // Freeing up the node for the inference run, if it is currently busy with another run.
    UnloadingModel {
        operation_id: OperationId,
        user_peer_id: PeerId,
        node_peer_id: PeerId,
        model_id: ModelId,
    },

    /// Loading the model on the node, if it is not already loaded.
    LoadingModel {
        operation_id: OperationId,
        user_peer_id: PeerId,
        node_peer_id: PeerId,
        model_id: ModelId,
    },

    /// The inference run is currently being processed by a node.
    InProgress {
        operation_id: OperationId,
        user_peer_id: PeerId,
        node_peer_id: PeerId,
        model_id: ModelId,
    },

    /// The inference run has completed successfully.
    Completed(OperationId),

    /// The inference run has failed with an error.
    Failed {
        operation_id: OperationId,
        error_message: String,
    },
}

/// A message sent from NexoGateway to the NexoAgent.
///
/// NOTE: These closely mirror the messages sent between the node, user and gateway, and
/// we try to reuse the same types where possible. This allows us to keep a strict protocol
/// here between NexoGateway and NexoAgent, but still remove a lot of boilerplate code
/// by reusing the same types.
#[derive(Debug, IntoStaticStr, PartialEq)]
pub enum NexoAgentInput {
    /// A new user has connected
    UserConnected(User),

    /// A new node has connected
    NodeConnected(Node),

    /// A user has disconnected
    UserDisconnected(PeerId),

    /// A node has disconnected
    NodeDisconnected(PeerId),

    /// A request to start a new inference run operation with the specified parameters.
    UserStartInferenceRun(InferenceIntent),

    /// A request to append additional instructions to an ongoing inference run operation.
    ///
    /// TODO: Review the required payload.
    UserAppendInferenceInstructions {
        /// The unique identifier for the inference operation to which the instructions should be appended.
        operation_id: OperationId,

        /// The additional instructions to be appended to the ongoing inference operation.
        instructions: InferenceIntent,
    },

    /// A request to compact a given session.
    UserCompact(CompactionRequest),

    /// An event emitted from the Node related to an inference run operation.
    NodeInferenceRunEvent(NexoEvent<InferenceRunEvent>),

    /// Retrieve the current state of the whole Nexo system.
    GetState {
        /// The requesting user peer that should receive the response.
        requester: PeerId,

        /// The operation identifier to preserve in the gateway response.
        operation_id: OperationId,
    },
}

/// An event sent from the Nexo Agent
///
/// NOTE: These closely mirror the messages sent between the node, user and gateway, and
/// we try to reuse the same types where possible. This allows us to keep a strict protocol
/// here between NexoGateway and NexoAgent, but still remove a lot of boilerplate code
/// by reusing the same types.
pub enum NexoAgentOutput {
    /// Send a fully prepared request to the node for processing.
    StartInferenceRun(Node, InferenceRequest),

    /// Return the current state of the Nexo system.
    GetState {
        /// The requesting user peer that should receive the response.
        requester: PeerId,

        /// The operation identifier to preserve in the gateway response.
        operation_id: OperationId,

        /// Snapshot of the current in-memory system state.
        state: NexoState,
    },
}

/// The Nexo Agent is responsible for coordinating a single session.
///
/// Bounding a NexoAgent to a single session allows us to enforce the append-only log of a session,
/// and manage the state of that session in a single place. If enough (inference, tools) resources are available,
/// we can run multiple sessions in parallel.
///
/// UPDATE: Again switching my thinking. I will have a single NexoAgent that uses start() to start a background loop
/// that will poll for work. That work will then be handled by a run. Serialization will be handled by state in the
/// database + transactions.
#[derive(Debug)]
pub struct NexoAgent {
    /// The Database client
    db: DbClient,

    /// The current state of the Nexo System
    state: NexoState,

    /// The work queue, tmp just a string
    fifo_queue: VecDeque<AgentJob>,
}

impl NexoAgent {
    /// Creates a new NexoAgent instance.
    pub fn new() -> Self {
        Self {
            fifo_queue: VecDeque::new(),
            state: NexoState::new(),
            db: DbClient::new(),
        }
    }

    /// Starts the NexoAgent.
    ///
    /// This is the main entry point and will start a loop that processes tasks from it's queue.
    pub fn start(
        mut self,
        mut input_rx: mpsc::Receiver<NexoAgentInput>,
        gateway_output_tx: mpsc::Sender<NexoAgentOutput>,
    ) -> tokio::task::JoinHandle<Result> {
        info!("Starting NexoAgent...");

        tokio::spawn(async move {
            let (agent_output_tx, mut agent_output_rx) = mpsc::channel(100);
            let mut queue_poll = time::interval(QUEUE_POLL_INTERVAL);

            loop {
                tokio::select! {
                    _ = queue_poll.tick() => {
                        self.process_fifo_queue_tick().await;
                    }

                    maybe_input = input_rx.recv() => {
                        let Some(input) = maybe_input else {
                            info!("NexoAgent input channel closed; stopping agent loop");
                            break;
                        };

                        self.handle_input(input, &agent_output_tx).await?;
                    }

                    maybe_output = agent_output_rx.recv() => {
                        let Some(output) = maybe_output else {
                            info!("NexoAgent deferred output channel closed; stopping agent loop");
                            break;
                        };

                        self.forward_output_to_gateway(output, &gateway_output_tx).await?;
                    }
                }
            }

            Ok(())
        })
    }

    /// Handles a NexoAgentInput sent to the NexoAgent.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to be handled by the NexoAgent.
    async fn handle_input(
        &mut self,
        input: NexoAgentInput,
        output_tx: &mpsc::Sender<NexoAgentOutput>,
    ) -> Result {
        match input {
            NexoAgentInput::UserConnected(user) => {
                if let Err(error) = self.state.add_user(user) {
                    warn!(error = %error, "Failed to add user to in-memory state");
                }
            }
            NexoAgentInput::NodeConnected(node) => {
                if let Err(error) = self.state.add_node(node) {
                    warn!(error = %error, "Failed to add node to in-memory state");
                }
            }
            NexoAgentInput::UserDisconnected(peer_id) => {
                self.state.remove_user(&peer_id);
            }
            NexoAgentInput::NodeDisconnected(peer_id) => {
                self.state.remove_node(&peer_id);
            }
            NexoAgentInput::UserStartInferenceRun(_request) => {
                info!("UserStartInferenceRun accepted by placeholder agent loop");
            }
            NexoAgentInput::UserAppendInferenceInstructions {
                operation_id: _,
                instructions: _,
            } => {
                info!("UserAppendInferenceInstructions accepted by placeholder agent loop");
            }
            NexoAgentInput::UserCompact(_request) => {
                info!("UserCompact accepted by placeholder agent loop");
            }
            NexoAgentInput::NodeInferenceRunEvent(_event) => {
                info!("NodeInferenceRunEvent accepted by placeholder agent loop");
            }
            NexoAgentInput::GetState {
                requester,
                operation_id,
            } => {
                output_tx
                    .send(NexoAgentOutput::GetState {
                        requester,
                        operation_id,
                        state: self.state.clone(),
                    })
                    .await?;
            }
        }

        Ok(())
    }

    async fn process_fifo_queue_tick(&mut self) {
        if let Some(task) = self.fifo_queue.pop_front() {
            match task {
                AgentJob::RunInference(_) => {
                    info!("Queued inference job placeholder popped");
                } // AgentJob::RunTool(_) => {
                  //     info!("Queued tool job placeholder popped");
                  // }
            }
        }
    }

    async fn forward_output_to_gateway(
        &self,
        output: NexoAgentOutput,
        gateway_output_tx: &mpsc::Sender<NexoAgentOutput>,
    ) -> Result {
        gateway_output_tx.send(output).await?;
        Ok(())
    }

    /// A single run of the agent, which will process any pending inference requests and manage model state.
    async fn inference_run(&self, request: InferenceIntent) -> Result {
        // Based on the inference request, determine which model to use, and which node to route the request to.
        // The router will block until the model is loaded on the node, and then return the routing
        // information for the inference request.
        let (model_id, node_id) = self.route_inference(&request).await?;

        // Transform the inference request into the full request that needs to be sent to the node,
        //including any context, system prompt, etc.
        let full_request = self.collect_context(&request, model_id).await?;

        todo!("Send full composed request gateway for distributing to node");
        Ok(())
    }

    /// A single tool call execution
    async fn tool_run(&self, request: ToolCall) -> Result {
        todo!("Determine tool to use (router)");
        todo!("Execute tool call (tool manager)");
        Ok(())
    }

    /// I probably want to move this to separate context manager.
    async fn collect_context(
        &self,
        intent: &InferenceIntent,
        model_id: ModelId,
    ) -> Result<InferenceRequest> {
        // - Load the system prompt for the model, if it exists.
        // - Load the context for the session, including any previous inference runs, tool calls, etc.
        // - Return the collected context to be used in the inference run.
        let system_prompt = ConversationMessage::new_system_prompt("You are a helpful assistant.");
        // TODO: Will we use the RoleStrategy/merge developer into system here, or will we do that
        // at the node level?
        let developer_prompt =
            ConversationMessage::new_developer_prompt("Do not do dangerous things.");

        let conversation_history = vec![ConversationMessage::new_text("Hello, how are you?")];

        let request = InferenceRequest::from_intent(
            intent,
            model_id,
            vec![
                vec![system_prompt],
                vec![developer_prompt],
                conversation_history,
            ]
            .concat(),
        );

        Ok(request)
    }

    async fn route_tool(&self, tool_call: &ToolCall) -> Result<Node> {
        // - Determine which node to use for the tool call.
        // - Return the routing information for the tool call.
        todo!("Implement tool routing logic");
    }

    /// I probably want to move this to separate routing manager.
    async fn route_inference(&self, request: &InferenceIntent) -> Result<(ModelId, Node)> {
        // - Determine which model to use for the inference request.
        // - Check if the model is already loaded on a node, or if it needs to be loaded. TODO: This needs
        //   to lock everything until we've got confirmation the model is loaded.
        // - Return the routing information for the inference request.
        todo!("Implement inference routing logic");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use nexo_core::{
        ClientInfo, DeviceInfo, InferenceOperation, ModelCapability, ModelSelection, OperationId,
        ReasoningSettings, SessionId, ToolChoice, UserProperties,
    };

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    #[tokio::test]
    async fn get_state_returns_current_snapshot() {
        let mut agent = NexoAgent::new();
        let user = test_user();
        let peer_id = user.id();

        agent
            .handle_input(
                NexoAgentInput::UserConnected(user),
                &tokio::sync::mpsc::channel(1).0,
            )
            .await
            .expect("failed to handle connected user");

        let operation_id = OperationId::new();
        let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(1);
        let output = agent
            .handle_input(
                NexoAgentInput::GetState {
                    requester: peer_id,
                    operation_id,
                },
                &output_tx,
            )
            .await;

        assert!(output.is_ok());

        let Some(NexoAgentOutput::GetState {
            requester,
            operation_id: out_operation_id,
            state,
        }) = output_rx.recv().await
        else {
            panic!("expected get_state output")
        };

        assert_eq!(requester, peer_id);
        assert_eq!(out_operation_id, operation_id);
        assert_eq!(state.user_count(), 1);
        assert_eq!(state.node_count(), 0);
    }

    #[tokio::test]
    async fn out_of_scope_inputs_do_not_panic_or_emit_output() {
        let mut agent = NexoAgent::new();
        let user = test_user();
        let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(1);

        let output = agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun(InferenceIntent {
                    operation_id: OperationId::new(),
                    session_id: SessionId::new(),
                    model_selection: ModelSelection::Capabilities(vec![
                        ModelCapability::TextGeneration,
                        ModelCapability::Streaming,
                    ]),
                    operation: InferenceOperation::MultiModal(MultiModalPayload::new_round(
                        vec![ConversationMessage::new_text("hello")],
                        Vec::new(),
                        ToolChoice::Automatic,
                        ReasoningSettings::default(),
                    )),
                }),
                &output_tx,
            )
            .await;
        assert!(output.is_ok());
        assert!(output_rx.try_recv().is_err());

        agent
            .handle_input(NexoAgentInput::UserConnected(user.clone()), &output_tx)
            .await
            .expect("failed to handle connected user");
        agent
            .handle_input(NexoAgentInput::UserDisconnected(user.id()), &output_tx)
            .await
            .expect("failed to handle disconnected user");

        agent
            .handle_input(
                NexoAgentInput::GetState {
                    requester: user.id(),
                    operation_id: OperationId::new(),
                },
                &output_tx,
            )
            .await
            .expect("failed to handle get_state");

        let Some(NexoAgentOutput::GetState { state, .. }) = output_rx.recv().await else {
            panic!("expected get_state output")
        };
        assert_eq!(state.user_count(), 0);
    }
}
