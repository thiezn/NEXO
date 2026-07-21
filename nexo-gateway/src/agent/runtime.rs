use super::messages::{NexoAgentInput, NexoAgentOutput};
use super::{AgentJobKind, InferenceRoutingCandidate};
use crate::Result;
use crate::memory::db::DbClient;
use nexo_core::{
    CompactionRequest, ConversationMessage, InferenceIntent, InferenceMeta, InferenceOutputDelta,
    InferenceRequest, InferenceRunEvent, LoadModelEvent, ModelDefinition, ModelId, ModelSelection,
    NexoState, OperationId, PeerId, StreamSeq, UnloadModelEvent,
};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio::time::{self, Duration};
use tracing::{debug, error, info, warn};

const QUEUE_POLL_INTERVAL: Duration = Duration::from_secs(5);
const JOB_WAIT_TIMEOUT: chrono::Duration = chrono::Duration::minutes(5);

/// The Nexo Agent is responsible for coordinating session work and persisted lifecycle updates.
#[derive(Debug)]
pub struct NexoAgent {
    /// The database client.
    db: DbClient,
    /// Resolved path to the git-backed nexo-storage repository configured for this runtime.
    nexo_storage_path: PathBuf,
    /// The current in-memory system state.
    state: NexoState,
}

impl NexoAgent {
    /// Create a new NexoAgent instance backed by the default database.
    pub fn new() -> Self {
        Self::with_db_and_storage(DbClient::new(), default_nexo_storage_path())
    }

    /// Create a new NexoAgent instance from resolved runtime config paths.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Resolved filesystem path to the SQLite database file.
    /// * `nexo_storage_path` - Resolved filesystem path to the git-backed storage root.
    pub fn from_config(db_path: &Path, nexo_storage_path: &Path) -> Result<Self> {
        let db = DbClient::from_path(db_path)?;
        Ok(Self::with_db_and_storage(
            db,
            nexo_storage_path.to_path_buf(),
        ))
    }

    /// Create a new NexoAgent instance backed by an injected database client.
    ///
    /// # Arguments
    ///
    /// * `db` - The database client to use for persistence.
    #[cfg(test)]
    pub(crate) fn with_db(db: DbClient) -> Self {
        Self::with_db_and_storage(db, default_nexo_storage_path())
    }

    fn with_db_and_storage(db: DbClient, nexo_storage_path: PathBuf) -> Self {
        Self {
            state: NexoState::new(),
            db,
            nexo_storage_path,
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
        info!(
            nexo_storage_path = %self.nexo_storage_path.display(),
            "Starting NexoAgent..."
        );

        tokio::spawn(async move {
            let (agent_output_tx, mut agent_output_rx) = mpsc::channel(100);
            let mut queue_poll = time::interval(QUEUE_POLL_INTERVAL);

            loop {
                tokio::select! {
                    _ = queue_poll.tick() => {
                        if let Err(error) = self.process_fifo_queue_tick(&agent_output_tx).await {
                            error!(error = %error, "Agent queue tick failed; stopping agent loop");
                            return Err(error);
                        }
                    }
                    maybe_input = input_rx.recv() => {
                        let Some(input) = maybe_input else {
                            info!("NexoAgent input channel closed; stopping agent loop");
                            break;
                        };

                        if let Err(error) = self.handle_input(input, &agent_output_tx).await {
                            error!(error = %error, "Agent input handling failed; stopping agent loop");
                            return Err(error);
                        }
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
                self.queue_user_start_inference_run(requester, intent)
                    .await?;
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
            NexoAgentInput::NodeLoadModelEvent {
                operation_id,
                source_node_id,
                event,
            } => match event {
                LoadModelEvent::Started { model_id } => {
                    debug!(%operation_id, %source_node_id, %model_id, "Node started loading model");
                }
                LoadModelEvent::Completed { model_id } => {
                    self.handle_model_loaded(operation_id, source_node_id, model_id)
                        .await?;
                }
                LoadModelEvent::Failed { model_id, error } => {
                    error!(%operation_id, %source_node_id, %model_id, error = %error, "Node failed to load model");
                }
            },
            NexoAgentInput::NodeUnloadModelEvent {
                operation_id,
                source_node_id,
                event,
            } => match event {
                UnloadModelEvent::Started { model_id } => {
                    debug!(%operation_id, %source_node_id, %model_id, "Node started unloading model");
                }
                UnloadModelEvent::Completed { model_id } => {
                    self.handle_model_unloaded(operation_id, source_node_id, model_id)
                        .await?;
                }
                UnloadModelEvent::Failed { model_id, error } => {
                    error!(%operation_id, %source_node_id, %model_id, error = %error, "Node failed to unload model");
                }
            },
            NexoAgentInput::NodeInferenceRunEvent {
                source_node_id,
                operation_id,
                event,
            } => {
                self.handle_node_inference_run_event(source_node_id, operation_id, event)
                    .await?;
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
        self.db.enqueue_inference_job(requester, &intent).await?;
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

    /// Advance a bounded FIFO batch of runnable persisted jobs by at most one external action each.
    ///
    /// # Arguments
    ///
    /// * `output_tx` - Domain-output channel used to request gateway-managed external actions.
    async fn process_fifo_queue_tick(
        &mut self,
        output_tx: &mpsc::Sender<NexoAgentOutput>,
    ) -> Result {
        for job in self.db.list_runnable_jobs().await? {
            match job.kind {
                AgentJobKind::RunInference => {
                    let intent = self.db.get_inference_intent(job.operation_id).await?;
                    self.progress_inference_job(
                        job.operation_id,
                        job.user_peer_id,
                        intent,
                        output_tx,
                    )
                    .await?;
                }
            }
        }

        Ok(())
    }

    /// Advance one runnable inference job according to its durable workflow state.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation being started.
    /// * `user_peer_id` - The owning user peer for the run.
    /// * `intent` - The persisted inference intent payload.
    /// * `output_tx` - Domain-output channel used to request the next gateway-managed action.
    async fn progress_inference_job(
        &mut self,
        operation_id: OperationId,
        user_peer_id: PeerId,
        intent: InferenceIntent,
        output_tx: &mpsc::Sender<NexoAgentOutput>,
    ) -> Result {
        let snapshot = self.db.load_inference_run_snapshot(operation_id).await?;
        match snapshot.state {
            super::InferenceRunState::Queued => {
                if self.db.begin_context_preparation(operation_id).await? {
                    info!(%operation_id, %user_peer_id, "Queued inference job entered preparing_context");
                }
            }
            super::InferenceRunState::PreparingContext => {
                let candidates = self.route_inference_candidates(&intent).await?;
                if candidates.is_empty() {
                    debug!(%operation_id, "No connected node can satisfy the inference model selection");
                }
                for candidate in candidates {
                    let deadline = (chrono::Utc::now() + JOB_WAIT_TIMEOUT).to_rfc3339();
                    let transitioned = match candidate {
                        InferenceRoutingCandidate::Loaded { node, model_id } => {
                            let transitioned = self
                                .db
                                .begin_inference_on_loaded_model(
                                    operation_id,
                                    node.id(),
                                    model_id,
                                    &deadline,
                                )
                                .await?;
                            if transitioned {
                                output_tx
                                    .send(NexoAgentOutput::StartInference {
                                        node_peer_id: node.id(),
                                        operation_id,
                                        request: self.collect_context(&intent, model_id).await?,
                                    })
                                    .await?;
                                info!(%operation_id, node_peer_id = %node.id(), %model_id, "Dispatched inference using loaded model");
                            }
                            transitioned
                        }
                        InferenceRoutingCandidate::Load { node, model_id } => {
                            let transitioned = self
                                .db
                                .begin_model_loading(operation_id, node.id(), model_id, &deadline)
                                .await?;
                            if transitioned {
                                output_tx
                                    .send(NexoAgentOutput::LoadModel {
                                        node_peer_id: node.id(),
                                        operation_id,
                                        model_id,
                                    })
                                    .await?;
                                info!(%operation_id, node_peer_id = %node.id(), %model_id, "Dispatched model load to empty node");
                            }
                            transitioned
                        }
                        InferenceRoutingCandidate::UnloadThenLoad {
                            node,
                            model_id,
                            unloading_model_id,
                        } => {
                            let transitioned = self
                                .db
                                .begin_model_unloading(
                                    operation_id,
                                    node.id(),
                                    model_id,
                                    unloading_model_id,
                                    &deadline,
                                )
                                .await?;
                            if transitioned {
                                output_tx
                                    .send(NexoAgentOutput::UnloadModel {
                                        node_peer_id: node.id(),
                                        operation_id,
                                        model_id: unloading_model_id,
                                    })
                                    .await?;
                                info!(%operation_id, node_peer_id = %node.id(), target_model_id = %model_id, %unloading_model_id, "Dispatched model eviction before target load");
                            }
                            transitioned
                        }
                    };
                    if transitioned {
                        break;
                    }
                    debug!(%operation_id, "Routing candidate became unavailable before its lease was acquired");
                }
            }
            super::InferenceRunState::UnloadingModel {
                node_peer_id,
                model_id,
                unloading_model_id,
            } => {
                let deadline = (chrono::Utc::now() + JOB_WAIT_TIMEOUT).to_rfc3339();
                if self
                    .db
                    .begin_model_loading_after_unload(
                        operation_id,
                        node_peer_id,
                        model_id,
                        unloading_model_id,
                        &deadline,
                    )
                    .await?
                {
                    output_tx
                        .send(NexoAgentOutput::LoadModel {
                            node_peer_id,
                            operation_id,
                            model_id,
                        })
                        .await?;
                    info!(%operation_id, %node_peer_id, %model_id, %unloading_model_id, "Dispatched target model load after eviction");
                }
            }
            super::InferenceRunState::LoadingModel {
                node_peer_id,
                model_id,
            } => {
                let request = self.collect_context(&intent, model_id).await?;
                let deadline = (chrono::Utc::now() + JOB_WAIT_TIMEOUT).to_rfc3339();
                if self
                    .db
                    .begin_inference_after_model_load(
                        operation_id,
                        node_peer_id,
                        model_id,
                        &deadline,
                    )
                    .await?
                {
                    output_tx
                        .send(NexoAgentOutput::StartInference {
                            node_peer_id,
                            operation_id,
                            request,
                        })
                        .await?;
                    info!(%operation_id, %node_peer_id, %model_id, "Dispatched inference after model load completion");
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle a node model-loaded notification.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation associated with the model load.
    /// * `source_node_id` - The authenticated node that loaded the model.
    /// * `model_id` - The model that was loaded.
    async fn handle_model_loaded(
        &mut self,
        operation_id: OperationId,
        source_node_id: PeerId,
        model_id: ModelId,
    ) -> Result {
        if self
            .db
            .complete_model_loading(operation_id, source_node_id, model_id)
            .await?
        {
            info!(%operation_id, node_peer_id = %source_node_id, %model_id, "Model load persisted; inference job is runnable");
        } else {
            warn!(%operation_id, node_peer_id = %source_node_id, %model_id, "Ignoring stale or mismatched model-loaded event");
        }
        Ok(())
    }

    /// Handle a node model-unloaded notification.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - The operation associated with the model unload.
    /// * `source_node_id` - The authenticated node that unloaded the model.
    /// * `model_id` - The model that was unloaded.
    async fn handle_model_unloaded(
        &mut self,
        operation_id: OperationId,
        source_node_id: PeerId,
        model_id: ModelId,
    ) -> Result {
        if self
            .db
            .complete_model_unloading(operation_id, source_node_id, model_id)
            .await?
        {
            info!(%operation_id, node_peer_id = %source_node_id, unloading_model_id = %model_id, "Model eviction persisted; inference job is runnable");
        } else {
            warn!(%operation_id, node_peer_id = %source_node_id, unloading_model_id = %model_id, "Ignoring stale or mismatched model-unloaded event");
        }
        Ok(())
    }

    /// Handle an authenticated, correlated node inference event.
    ///
    /// # Arguments
    ///
    /// * `source_node_id` - Authenticated node that emitted the event.
    /// * `operation_id` - Operation correlated by the gateway protocol layer.
    /// * `event` - Shared inference lifecycle event payload.
    async fn handle_node_inference_run_event(
        &mut self,
        source_node_id: PeerId,
        operation_id: OperationId,
        event: InferenceRunEvent,
    ) -> Result {
        debug!(%operation_id, %source_node_id, "Handling authenticated inference event");
        self.handle_correlated_inference_run_event(operation_id, event)
            .await
    }

    /// Apply one shared inference lifecycle event to its durable operation.
    ///
    /// # Arguments
    ///
    /// * `operation_id` - Operation associated with the event.
    /// * `event` - Shared inference lifecycle event.
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
                self.handle_inference_round_completed(operation_id, meta)
                    .await
            }
            InferenceRunEvent::Output { meta, seq, output } => {
                self.handle_inference_output(operation_id, meta, seq, output)
                    .await
            }
            InferenceRunEvent::RunCompleted {
                meta,
                total_outputs,
            } => {
                self.handle_inference_run_completed(operation_id, meta, total_outputs)
                    .await
            }
            InferenceRunEvent::Cancelled { meta, reason } => {
                self.handle_inference_cancelled(operation_id, meta, reason)
                    .await
            }
            InferenceRunEvent::Failed { meta, error } => {
                self.handle_inference_failed(operation_id, meta, error)
                    .await
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
            [
                vec![system_prompt],
                vec![developer_prompt],
                conversation_history,
            ]
            .concat(),
        );

        Ok(request)
    }

    /// Build ordered node/model routes for an inference request.
    ///
    /// Loaded models are preferred, followed by empty nodes. Occupied nodes that require one
    /// model eviction are returned last because the current node contract does not expose memory
    /// capacity or per-model memory requirements.
    ///
    /// # Arguments
    ///
    /// * `request` - Persisted inference intent whose model selection must be satisfied.
    async fn route_inference_candidates(
        &self,
        request: &InferenceIntent,
    ) -> Result<Vec<InferenceRoutingCandidate>> {
        let mut candidates = self
            .db
            .list_nodes()
            .await?
            .into_iter()
            .filter(|node| self.state.nodes().contains_key(&node.id()))
            .flat_map(|node| {
                node.models_on_disk()
                    .iter()
                    .filter(|model_id| {
                        model_matches_selection(**model_id, &request.model_selection)
                    })
                    .map(|model_id| {
                        if node.models_in_memory().contains(model_id) {
                            InferenceRoutingCandidate::Loaded {
                                node: node.clone(),
                                model_id: *model_id,
                            }
                        } else if node.models_in_memory().is_empty() {
                            InferenceRoutingCandidate::Load {
                                node: node.clone(),
                                model_id: *model_id,
                            }
                        } else {
                            let unloading_model_id = node
                                .models_in_memory()
                                .iter()
                                .copied()
                                .min_by_key(|loaded_model_id| String::from(*loaded_model_id))
                                .expect("occupied node has at least one loaded model");
                            InferenceRoutingCandidate::UnloadThenLoad {
                                node: node.clone(),
                                model_id: *model_id,
                                unloading_model_id,
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(InferenceRoutingCandidate::sort_key);

        Ok(candidates)
    }
}

fn model_matches_selection(model_id: ModelId, selection: &ModelSelection) -> bool {
    match selection {
        ModelSelection::SpecificModel(selected) => *selected == model_id,
        ModelSelection::Capabilities(required) => {
            let definition = ModelDefinition::new(model_id);
            required
                .iter()
                .all(|capability| definition.capabilities().contains(capability))
        }
    }
}

fn default_nexo_storage_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nexo")
        .join("nexo-storage")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::agent::InferenceRunState;
    use nexo_core::inference::requests::multimodal::MultiModalPayload;
    use nexo_core::{
        ClientInfo, DeviceInfo, InferenceOperation, ModelCapability, ModelSelection, Node,
        NodeProperties, NodeState, OperationId, ReasoningSettings, SessionId, ToolChoice, User,
        UserProperties,
    };
    use sqlx::sqlite::SqlitePoolOptions;
    use std::collections::HashSet;

    fn test_user() -> User {
        let properties =
            UserProperties::new(ClientInfo::new("test-user"), DeviceInfo::default(), "token");
        User::from_properties(&properties)
    }

    fn test_node(models_in_memory: HashSet<ModelId>) -> Node {
        let properties =
            NodeProperties::builder(ClientInfo::new("test-node"), DeviceInfo::default(), "token")
                .models(vec![ModelId::Gemma4E4bItUqffAfq6])
                .build();
        Node::from_properties(&properties, NodeState::Idle, models_in_memory)
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
        let jobs = db
            .list_runnable_jobs()
            .await
            .expect("failed to load runnable jobs");
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].operation_id, operation_id);
    }

    #[tokio::test]
    async fn sqlite_queue_survives_agent_reconstruction() {
        let db = test_db().await;
        let mut accepting_agent = NexoAgent::with_db(db.clone());
        let user = test_user();
        let intent = test_intent();
        let operation_id = intent.operation_id;
        let (output_tx, _output_rx) = tokio::sync::mpsc::channel(1);

        accepting_agent
            .handle_input(NexoAgentInput::UserConnected(user.clone()), &output_tx)
            .await
            .unwrap();
        accepting_agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun {
                    requester: user.id(),
                    intent,
                },
                &output_tx,
            )
            .await
            .unwrap();
        drop(accepting_agent);

        let mut reconstructed_agent = NexoAgent::with_db(db.clone());
        reconstructed_agent
            .process_fifo_queue_tick(&output_tx)
            .await
            .unwrap();
        reconstructed_agent
            .process_fifo_queue_tick(&output_tx)
            .await
            .unwrap();

        let snapshot = db.load_inference_run_snapshot(operation_id).await.unwrap();
        assert_eq!(snapshot.state, InferenceRunState::PreparingContext);
    }

    #[tokio::test]
    async fn model_loaded_only_wakes_job_until_next_tick() {
        let db = test_db().await;
        let mut agent = NexoAgent::with_db(db.clone());
        let user = test_user();
        let node = test_node(HashSet::new());
        let intent = test_intent();
        let operation_id = intent.operation_id;
        let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(4);

        agent
            .handle_input(NexoAgentInput::UserConnected(user.clone()), &output_tx)
            .await
            .unwrap();
        agent
            .handle_input(NexoAgentInput::NodeConnected(node.clone()), &output_tx)
            .await
            .unwrap();
        agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun {
                    requester: user.id(),
                    intent,
                },
                &output_tx,
            )
            .await
            .unwrap();

        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        let Some(NexoAgentOutput::LoadModel {
            node_peer_id,
            operation_id: dispatched_operation_id,
            model_id,
        }) = output_rx.recv().await
        else {
            panic!("expected model-load command")
        };
        assert_eq!(node_peer_id, node.id());
        assert_eq!(dispatched_operation_id, operation_id);
        assert_eq!(model_id, ModelId::Gemma4E4bItUqffAfq6);

        agent
            .handle_input(
                NexoAgentInput::NodeLoadModelEvent {
                    operation_id,
                    source_node_id: node.id(),
                    event: LoadModelEvent::Completed {
                        model_id: ModelId::Gemma4E4bItUqffAfq6,
                    },
                },
                &output_tx,
            )
            .await
            .unwrap();
        assert!(output_rx.try_recv().is_err());

        let loading = db.load_inference_run_snapshot(operation_id).await.unwrap();
        assert!(matches!(
            loading.state,
            InferenceRunState::LoadingModel { .. }
        ));

        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        let Some(NexoAgentOutput::StartInference {
            node_peer_id,
            operation_id: dispatched_operation_id,
            ..
        }) = output_rx.recv().await
        else {
            panic!("expected inference command")
        };
        assert_eq!(node_peer_id, node.id());
        assert_eq!(dispatched_operation_id, operation_id);

        let running = db.load_inference_run_snapshot(operation_id).await.unwrap();
        assert!(matches!(
            running.state,
            InferenceRunState::InProgress { .. }
        ));
    }

    #[tokio::test]
    async fn occupied_node_unloads_before_loading_target_model() {
        let db = test_db().await;
        let mut agent = NexoAgent::with_db(db.clone());
        let user = test_user();
        let unloading_model_id = ModelId::Kokoro82m;
        let target_model_id = ModelId::Gemma4E4bItUqffAfq6;
        let node = test_node(HashSet::from([unloading_model_id]));
        let intent = test_intent();
        let operation_id = intent.operation_id;
        let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(8);

        agent
            .handle_input(NexoAgentInput::UserConnected(user.clone()), &output_tx)
            .await
            .unwrap();
        agent
            .handle_input(NexoAgentInput::NodeConnected(node.clone()), &output_tx)
            .await
            .unwrap();
        agent
            .handle_input(
                NexoAgentInput::UserStartInferenceRun {
                    requester: user.id(),
                    intent,
                },
                &output_tx,
            )
            .await
            .unwrap();

        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        assert!(matches!(
            output_rx.recv().await,
            Some(NexoAgentOutput::UnloadModel {
                node_peer_id,
                operation_id: dispatched_operation_id,
                model_id,
            }) if node_peer_id == node.id()
                && dispatched_operation_id == operation_id
                && model_id == unloading_model_id
        ));

        agent
            .handle_input(
                NexoAgentInput::NodeUnloadModelEvent {
                    operation_id,
                    source_node_id: node.id(),
                    event: UnloadModelEvent::Completed {
                        model_id: unloading_model_id,
                    },
                },
                &output_tx,
            )
            .await
            .unwrap();
        assert!(output_rx.try_recv().is_err());

        let unloading = db.load_inference_run_snapshot(operation_id).await.unwrap();
        assert_eq!(
            unloading.state,
            InferenceRunState::UnloadingModel {
                node_peer_id: node.id(),
                model_id: target_model_id,
                unloading_model_id,
            }
        );

        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        assert!(matches!(
            output_rx.recv().await,
            Some(NexoAgentOutput::LoadModel {
                node_peer_id,
                operation_id: dispatched_operation_id,
                model_id,
            }) if node_peer_id == node.id()
                && dispatched_operation_id == operation_id
                && model_id == target_model_id
        ));

        agent
            .handle_input(
                NexoAgentInput::NodeLoadModelEvent {
                    operation_id,
                    source_node_id: node.id(),
                    event: LoadModelEvent::Completed {
                        model_id: target_model_id,
                    },
                },
                &output_tx,
            )
            .await
            .unwrap();
        assert!(output_rx.try_recv().is_err());

        agent.process_fifo_queue_tick(&output_tx).await.unwrap();
        assert!(matches!(
            output_rx.recv().await,
            Some(NexoAgentOutput::StartInference {
                node_peer_id,
                operation_id: dispatched_operation_id,
                ..
            }) if node_peer_id == node.id() && dispatched_operation_id == operation_id
        ));
    }
}
