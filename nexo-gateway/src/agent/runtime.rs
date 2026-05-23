//! Background agent task spawning and command transport.

use crate::server::state::SharedState;
use nexo_ws_schema::Frame;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};

/// Commands sent to the background agent task.
pub enum AgentCommand {
    /// Start or resume an agent run.
    RunAgent {
        /// Unique run identifier for the background task invocation.
        run_id: String,
        /// Session that owns the run.
        session_id: String,
        /// User prompt that started the run.
        prompt: String,
        /// Optional structured context appended to the request.
        context: Option<serde_json::Value>,
        /// Originating peer that submitted the run.
        peer_id: String,
        /// Explicit model requested for the run, if any.
        model_id: Option<String>,
        /// Optional stored context collection selected for the run.
        prefill_collection_id: Option<String>,
        /// Whether the model should expose thinking output.
        thinking: bool,
    },
    /// Drain queued runs when a compatible node becomes available.
    DrainQueue,
}

/// Handle used by request handlers to submit work to the agent task.
#[derive(Clone)]
pub struct AgentHandle {
    cmd_tx: mpsc::Sender<AgentCommand>,
}

impl AgentHandle {
    /// Spawn the background agent task and return a handle to it.
    pub fn spawn(db: SqlitePool, state: SharedState, event_tx: broadcast::Sender<Frame>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        tokio::spawn(agent_task(cmd_rx, db, state, event_tx));
        Self { cmd_tx }
    }

    /// Submit an agent command to the background task.
    pub async fn submit(
        &self,
        cmd: AgentCommand,
    ) -> Result<(), mpsc::error::SendError<AgentCommand>> {
        self.cmd_tx.send(cmd).await
    }
}

/// Process agent commands until the sender side is dropped.
async fn agent_task(
    mut cmd_rx: mpsc::Receiver<AgentCommand>,
    db: SqlitePool,
    state: SharedState,
    event_tx: broadcast::Sender<Frame>,
) {
    tracing::info!("Agent brain started");
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            AgentCommand::RunAgent {
                run_id,
                session_id,
                prompt,
                context,
                peer_id,
                model_id,
                prefill_collection_id,
                thinking,
            } => {
                tracing::info!(
                    "Starting agent run {run_id} (session={session_id}, thinking={thinking})"
                );
                super::r#loop::start_run(
                    &run_id,
                    &session_id,
                    &prompt,
                    context.as_ref(),
                    &peer_id,
                    &db,
                    &state,
                    &event_tx,
                    model_id.as_deref(),
                    prefill_collection_id.as_deref(),
                    thinking,
                )
                .await;
            }
            AgentCommand::DrainQueue => {
                tracing::info!("Queue drain triggered");
                super::queue::drain_queue(&db, &state, &event_tx).await;
            }
        }
    }
    tracing::info!("Agent brain shut down");
}
