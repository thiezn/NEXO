use crate::Result;
use crate::memory::db::DbClient;
use nexo_core::{
    CompactionRequest, ConversationMessage, InferenceIntent, InferenceRequest, ModelId, NexoState,
    Node, PeerId, ToolCall, User,
};
use nexo_ws_schema::{InferenceRunEvent, NexoEvent};
use std::collections::VecDeque;
use std::thread::sleep;
use std::time::Duration;
use strum::IntoStaticStr;
use tracing::info;

/// A single job that the NexoAgent can perform.
///
/// These jobs are queued up in the NexoAgent queue. The agent is responsible
/// for handling parallelism/sequencing.
#[derive(Debug, IntoStaticStr, PartialEq)]
enum AgentJob {
    /// A job to run an inference request.
    RunInference(InferenceIntent),

    /// A job to run a tool call.
    RunTool(ToolCall),
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

    // A new node has connected
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
        operation_id: nexo_core::OperationId,

        /// The additional instructions to be appended to the ongoing inference operation.
        instructions: InferenceIntent,
    },

    /// A request to compact a given session.
    UserCompact(CompactionRequest),

    /// An event emitted from the Node related to an inference run operation.
    NodeInferenceRunEvent(NexoEvent<InferenceRunEvent>),

    /// Retrieve the current state of the whole Nexo system
    GetState,
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

    /// Return the current state of the Nexo system
    GetState(NexoState),
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

    /// The current state of the Nexo Sytem
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
    pub fn start(mut self) {
        info!("Starting NexoAgent...");

        // TODO: This is just a temporary loop and queue. We need
        // logic to determine what the actual next task is that we're
        // allowed to execute.
        //
        // Some points to consider
        // - We cannot run a new inference run on the same same session
        // - Queue needs to be stored in persistent storage so we can recover from crash/reboot
        // - We can probably solve this in SQL with a fancy query, if we can somehow lock a session there. It would
        //   be nice if all that logic is inside the DB, to avoid having to handle this in code.
        // - Recoverability will need some thought. Need towrite out some scenarios like, gateway crash, node crash,
        //   BOTH gateway and node crashes. Especially the latter case is important as the gateway might boot up again
        //   and assume the node is still working on an inference request. However, if the node also crashed, we will
        //   never get response for it so need some way to detect that/get node state perhaps, etc.
        loop {
            if let Some(task) = self.fifo_queue.pop_front() {
                match task {
                    AgentJob::RunInference(_) => {
                        info!("Processing inference run");
                    }
                    AgentJob::RunTool(_) => {
                        info!("Processing tool call");
                    }
                }
            }
            sleep(Duration::from_secs(5));
        }
    }

    /// Handles a command sent to the NexoAgent.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to be handled by the NexoAgent.
    async fn handle_command(&mut self, command: NexoAgentInput) -> Result {
        match command {
            NexoAgentInput::UserConnected(user) => {
                self.state.add_user(user)?;
            }
            NexoAgentInput::NodeConnected(node) => {
                self.state.add_node(node)?;
            }
            NexoAgentInput::UserDisconnected(peer_id) => {
                self.state.remove_user(&peer_id);
            }
            NexoAgentInput::NodeDisconnected(peer_id) => {
                self.state.remove_node(&peer_id);
            }
            NexoAgentInput::UserStartInferenceRun(request) => {
                todo!("Store in queue")
            }
            NexoAgentInput::UserAppendInferenceInstructions {
                operation_id,
                instructions,
            } => {
                todo!("Store in queue")
            }
            NexoAgentInput::UserCompact(request) => {
                todo!("Store in queue")
            }
            NexoAgentInput::NodeInferenceRunEvent(event) => {
                todo!("Store in queue")
            }
            NexoAgentInput::GetState => {
                todo!("Return current state")
            }
        }
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

        Ok(Node::new())
    }

    /// I probably want to move this to separate routing manager.
    async fn route_inference(&self, request: &InferenceIntent) -> Result<(ModelId, Node)> {
        // - Determine which model to use for the inference request.
        // - Check if the model is already loaded on a node, or if it needs to be loaded. TODO: This needs
        //   to lock everything until we've got confirmation the model is loaded.
        // - Return the routing information for the inference request.

        Ok((ModelId::Gemma4E4bItUqffAfq6, Node::new()))
    }
}
