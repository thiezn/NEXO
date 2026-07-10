use super::job::AgentJob;
use super::messages::{NexoAgentInput, NexoAgentOutput};
use super::InferenceRun;
use crate::memory::db::DbClient;
use crate::Result;
use nexo_core::{
    CompactionRequest, ConversationMessage, InferenceIntent, InferenceMeta, InferenceOutputDelta,
    InferenceRequest, ModelId, NexoState, Node, OperationId, PeerId, StreamSeq, ToolCall,
};
use nexo_ws_schema::{InferenceRunEvent, NexoEvent};
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tokio::time::{self, Duration};
use tracing::{info, warn};

const QUEUE_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// The Nexo Agent is responsible for coordinating session work and persisted lifecycle updates.
#[derive(Debug)]
pub struct NexoAgent {
    /// The database client.
    db: DbClient,
    /// The current in-memory system state.
    state: NexoState,
    /// The in-memory FIFO queue of pending jobs.
    pub(crate) fifo_queue: VecDeque<AgentJob>,
}

impl NexoAgent {
    /// Create a new NexoAgent instance backed by the default database.
    pub fn new() -> Self {
        Self::with_db(DbClient::new())
    }

    /// Create a new NexoAgent instance backed by an injected database client.
    ///
    /// # Arguments
    ///
    /// * `db` - The database client to use for persistence.
    pub(crate) fn with_db(db: DbClient) -> Self {
        Self {
            fifo_queue: VecDeque::new(),
            state: NexoState::new(),
            db,
        }
    }

    /// Start the background NexoAgent loop.
    ///
    /// # Arguments
    ///
    /// * `input_rx` - The receiver for agent input messages.
    /// * `gateway_output_tx` - The sender used to forward agent outputs back to the gateway.
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
                        self.process_fifo_queue_tick().await?;
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

                        gateway_output_tx.send(output).await?;
                    }
                }
            }

            Ok(())
        })
    }

    /// Handle a NexoAgent input message.
    ///
    /// # Arguments
    ///
    /// * `input` - The input message to process.
    /// * `output_tx` - The sender used for deferred agent outputs.
    pub(crate) async fn handle_input(
        &mut self,
        input: NexoAgentInput,
        output_tx: &mpsc::Sender<NexoAgentOutput>,
    ) -> Result {
        match input {
            NexoAgentInput::UserConnected(user) => {
                self.db.connect_user(&user).await?;
                if let Err(error) = self.state.add_user(user) {
                    warn!(error = %error, "Failed to add user to in-memory state");
                }
            }
            NexoAgentInput::NodeConnected(node) => {
                self.db.connect_node(&node).await?;
                if let Err(error) = self.state.add_node(node) {
                    warn!(error = %error, "Failed to add node to in-memory state");
                }
            }
            NexoAgentInput::UserDisconnected(peer_id) => {
                self.db.disconnect_user(peer_id).await?;
                self.state.remove_user(&peer_id);
            }
            NexoAgentInput::NodeDisconnected(peer_id) => {
                self.db.disconnect_node(peer_id).await?;
                self.state.remove_node(&peer_id);
            }
            NexoAgentInput::UserStartInferenceRun { requester, intent } => {
                self.queue_user_start_inference_run(requester, intent).await?;
            }
            NexoAgentInput::UserAppendInferenceInstructions {
                operation_id,
                instructions,
            } => {
                self.handle_user_append_inference_instructions(operation_id, instructions)
                    .await?;
            }
            NexoAgentInput::UserCompact(request) => {
                self.handle_user_compact(request).await?;
            }
            NexoAgentInput::ModelLoaded {
                operation_id,
                node,
                model_id,
            } => {
                self.handle_model_loaded(operation_id, node, model_id).await?;
            }
            NexoAgentInput::ModelUnloaded {
                operation_id,
                node,
                model_id,
            } => {
                self.handle_model_unloaded(operation_id, node, model_id).await?;
            }
            NexoAgentInput::NodeInferenceRunEvent(event) => {
                self.handle_node_inference_run_event(event).await?;
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

    /// Persist and enqueue a newly accepted user inference request.
    ///
    /// # Arguments
    ///
    /// * `requester` - The user peer that owns the requested operation.
    /// * `intent` - The inference intent to persist and enqueue.
    async fn queue_user_start_inference_run(
        &mut self,
        requester: PeerId,
        intent: InferenceIntent,
    ) -> Result {
        let operation_id = intent.operation_id;
        self.db.create_operation(operation_id, requester).await?;
        self.db.upsert_inference_intent(&intent).await?;

        let queued_run = InferenceRun::new(operation_id, requester);
        self.db.save_inference_run(&queued_run).await?;

        self.fifo_queue.push_back(AgentJob::from((requester, intent)));
        info!(%operation_id, %requester, "Queued inference run");
        Ok(())
    }

    /// Handle a request to append instructions to an existing inference operation.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation receiving the additional instructions.
    /// * `instructions` - The instructions to append to the operation.
    async fn handle_user_append_inference_instructions(
        &mut self,
        operation_id: OperationId,
        instructions: InferenceIntent,
    ) -> Result {
        let _ = (operation_id, instructions);
        info!("UserAppendInferenceInstructions accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle a user compaction request.
    ///
    /// # Arguments
    ///
    /// * `request` - The compaction request to process.
    async fn handle_user_compact(&mut self, request: CompactionRequest) -> Result {
        let _ = request;
        info!("UserCompact accepted by placeholder agent loop");
        Ok(())
    }

    /// Process the next queued job, if any.
    async fn process_fifo_queue_tick(&mut self) -> Result {
        if let Some(task) = self.fifo_queue.pop_front() {
            match task {
                AgentJob::RunInference {
                    operation_id,
                    user_peer_id,
                    intent,
                } => {
                    self.start_queued_inference_job(operation_id, user_peer_id, intent)
                        .await?;
                }
            }
        }

        Ok(())
    }

    /// Begin processing a queued inference job.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation being started.
    /// * `user_peer_id` - The owning user peer for the run.
    /// * `intent` - The persisted inference intent payload.
    async fn start_queued_inference_job(
        &self,
        operation_id: OperationId,
        user_peer_id: PeerId,
        intent: InferenceIntent,
    ) -> Result {
        let queued_run = InferenceRun::new(operation_id, user_peer_id);
        let preparing_run = queued_run.into_preparing_context();
        self.db.save_inference_run(&preparing_run).await?;

        let _ = intent;
        info!(%operation_id, %user_peer_id, "Queued inference job entered preparing_context");
        Ok(())
    }

    /// Handle a node model-loaded notification.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation associated with the model load.
    /// * `node` - The node that loaded the model.
    /// * `model_id` - The model that was loaded.
    async fn handle_model_loaded(
        &mut self,
        operation_id: OperationId,
        node: Node,
        model_id: ModelId,
    ) -> Result {
        let _ = (operation_id, node, model_id);
        // TODO: Resolve the persisted run for `operation_id`, confirm it is in `loading_model`,
        // and transition the typestate into `in_progress` once node/gateway routing is wired.
        info!("ModelLoaded accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle a node model-unloaded notification.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation associated with the model unload.
    /// * `node` - The node that unloaded the model.
    /// * `model_id` - The model that was unloaded.
    async fn handle_model_unloaded(
        &mut self,
        operation_id: OperationId,
        node: Node,
        model_id: ModelId,
    ) -> Result {
        let _ = (operation_id, node, model_id);
        // TODO: Resolve the persisted run for `operation_id`, confirm it is in `unloading_model`,
        // and transition the typestate into `loading_model` once load/unload orchestration is wired.
        info!("ModelUnloaded accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle a node-originated inference run event.
    ///
    /// # Arguments
    ///
    /// * `event` - The correlated or unsolicited inference run event to normalize.
    async fn handle_node_inference_run_event(
        &mut self,
        event: NexoEvent<InferenceRunEvent>,
    ) -> Result {
        match event {
            NexoEvent::Correlated {
                operation_id,
                event,
            } => self.handle_correlated_inference_run_event(operation_id, event).await,
            NexoEvent::Unsolicited { event } => {
                let _ = event;
                // TODO: Decide whether unsolicited inference events should be ignored, logged,
                // or mapped onto a durable run lookup keyed by `InferenceMeta` once node resume exists.
                info!("Unsolicited NodeInferenceRunEvent accepted by placeholder agent loop");
                Ok(())
            }
        }
    }

    /// Handle a correlated node-originated inference run event.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation associated with the event.
    /// * `event` - The specific inference run event payload.
    async fn handle_correlated_inference_run_event(
        &mut self,
        operation_id: OperationId,
        event: InferenceRunEvent,
    ) -> Result {
        match event {
            InferenceRunEvent::RunStarted { meta } => {
                self.handle_inference_run_started(operation_id, meta).await
            }
            InferenceRunEvent::RoundCompleted { meta } => {
                self.handle_inference_round_completed(operation_id, meta).await
            }
            InferenceRunEvent::Output { meta, seq, output } => {
                self.handle_inference_output(operation_id, meta, seq, output).await
            }
            InferenceRunEvent::RunCompleted {
                meta,
                total_outputs,
            } => self
                .handle_inference_run_completed(operation_id, meta, total_outputs)
                .await,
            InferenceRunEvent::Cancelled { meta, reason } => {
                self.handle_inference_cancelled(operation_id, meta, reason).await
            }
            InferenceRunEvent::Failed { meta, error } => {
                self.handle_inference_failed(operation_id, meta, error).await
            }
        }
    }

    /// Handle the node notification that an inference run started.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation that started.
    /// * `meta` - The execution metadata reported by the node.
    async fn handle_inference_run_started(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
    ) -> Result {
        let _ = (operation_id, meta);
        // TODO: Normalize `InferenceMeta` into the selected node/model pair and transition the
        // persisted run into `in_progress` using the typestate-backed persistence API.
        info!("Inference RunStarted accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle the node notification that an inference round completed.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation that advanced.
    /// * `meta` - The execution metadata reported by the node.
    async fn handle_inference_round_completed(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
    ) -> Result {
        let _ = (operation_id, meta);
        // TODO: Decide what durable round-progress data belongs alongside the run typestate and
        // record it once the gateway persists per-round inference progress.
        info!("Inference RoundCompleted accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle a streaming inference output event.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation emitting output.
    /// * `meta` - The execution metadata reported by the node.
    /// * `seq` - The output sequence number.
    /// * `output` - The output delta payload.
    async fn handle_inference_output(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
        seq: StreamSeq,
        output: InferenceOutputDelta,
    ) -> Result {
        let _ = (operation_id, meta, seq, output);
        // TODO: Persist or forward output deltas through the run pipeline once transcript/output
        // storage is integrated with the new inference run typestate flow.
        info!("Inference Output accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle the node notification that an inference run completed.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The completed operation.
    /// * `meta` - The execution metadata reported by the node.
    /// * `total_outputs` - The total number of output messages emitted by the node.
    async fn handle_inference_run_completed(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
        total_outputs: StreamSeq,
    ) -> Result {
        let _ = (operation_id, meta, total_outputs);
        // TODO: Rebuild a concrete active typestate from the persisted snapshot and transition it
        // into `completed` when node metadata wiring exists.
        info!("Inference RunCompleted accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle the node notification that an inference run was cancelled.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The cancelled operation.
    /// * `meta` - The execution metadata reported by the node.
    /// * `reason` - The optional cancellation reason from the node.
    async fn handle_inference_cancelled(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
        reason: Option<String>,
    ) -> Result {
        let _ = (operation_id, meta, reason);
        // TODO: Add an explicit cancelled inference run state or translate cancellation into a
        // durable failed/completed policy once product semantics are defined.
        info!("Inference Cancelled accepted by placeholder agent loop");
        Ok(())
    }

    /// Handle the node notification that an inference run failed.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The failed operation.
    /// * `meta` - The execution metadata reported by the node.
    /// * `error` - The failure message from the node.
    async fn handle_inference_failed(
        &mut self,
        operation_id: OperationId,
        meta: InferenceMeta,
        error: String,
    ) -> Result {
        let _ = (operation_id, meta, error);
        // TODO: Rebuild the active typestate from persisted run data and transition it into
        // `failed`, preserving any selected node/model from the current snapshot.
        info!("Inference Failed accepted by placeholder agent loop");
        Ok(())
    }

    /// Execute a single inference run.
    ///
    /// # Arguments
    ///
    /// * `request` - The inference intent to execute.
    async fn inference_run(&self, request: InferenceIntent) -> Result {
        let (model_id, node_id) = self.route_inference(&request).await?;
        let full_request = self.collect_context(&request, model_id).await?;

        let _ = (node_id, full_request);
        todo!("Send full composed request gateway for distributing to node");
    }

    /// Execute a single tool call.
    ///
    /// # Arguments
    ///
    /// * `request` - The tool call to execute.
    async fn tool_run(&self, request: ToolCall) -> Result {
        let _ = request;
        todo!("Determine tool to use (router)");
    }

    /// Collect the full inference context for a request.
    ///
    /// # Arguments
    ///
    /// * `intent` - The inference intent being prepared.
    /// * `model_id` - The resolved model identifier for the run.
    async fn collect_context(
        &self,
        intent: &InferenceIntent,
        model_id: ModelId,
    ) -> Result<InferenceRequest> {
        let system_prompt = ConversationMessage::new_system_prompt("You are a helpful assistant.");
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

    /// Determine which node should execute a tool call.
    ///
    /// # Arguments
    ///
    /// * `tool_call` - The tool call to route.
    async fn route_tool(&self, tool_call: &ToolCall) -> Result<Node> {
        let _ = tool_call;
        todo!("Implement tool routing logic");
    }

    /// Determine which node and model should execute an inference request.
    ///
    /// # Arguments
    ///
    /// * `request` - The inference request to route.
    async fn route_inference(&self, request: &InferenceIntent) -> Result<(ModelId, Node)> {
        let _ = request;
        todo!("Implement inference routing logic");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::agent::InferenceRunState;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use nexo_core::{
        ClientInfo, DeviceInfo, InferenceOperation, ModelCapability, ModelSelection, OperationId,
        ReasoningSettings, SessionId, ToolChoice, User, UserProperties,
    };
    use sqlx::sqlite::SqlitePoolOptions;

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    async fn test_db() -> DbClient {
        let pool = SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let db = DbClient::from_pool(pool);
        db.initialize_schema().await.unwrap();
        db
    }

    fn test_intent() -> InferenceIntent {
        InferenceIntent {
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
        }
    }

    #[tokio::test]
    async fn get_state_returns_current_snapshot() {
        let mut agent = NexoAgent::with_db(test_db().await);
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
        let mut agent = NexoAgent::with_db(test_db().await);
        let user = test_user();
        let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(1);

        let output = agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun {
                    requester: user.id(),
                    intent: test_intent(),
                },
                &output_tx,
            )
            .await;
        assert!(output.is_err());
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

    #[tokio::test]
    async fn start_inference_run_persists_queued_state_and_enqueues_job() {
        let db = test_db().await;
        let mut agent = NexoAgent::with_db(db.clone());
        let user = test_user();
        let intent = test_intent();
        let operation_id = intent.operation_id;

        agent
            .handle_input(
                NexoAgentInput::UserConnected(user.clone()),
                &tokio::sync::mpsc::channel(1).0,
            )
            .await
            .expect("failed to persist connected user");

        agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun {
                    requester: user.id(),
                    intent,
                },
                &tokio::sync::mpsc::channel(1).0,
            )
            .await
            .expect("failed to persist inference run");

        let snapshot = db
            .load_inference_run_snapshot(operation_id)
            .await
            .expect("failed to load persisted run snapshot");

        assert_eq!(snapshot.operation_id, operation_id);
        assert_eq!(snapshot.state, InferenceRunState::Queued);
        assert_eq!(agent.fifo_queue.len(), 1);
    }
}
