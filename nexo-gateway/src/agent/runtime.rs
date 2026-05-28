//! Background run task spawning and command transport.

use crate::server::state::SharedState;
use nexo_core::ReasoningSettings;
use nexo_ws_schema::Frame;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};

/// Commands sent to the background run task.
pub enum RunCommand {
    /// Start or resume a run.
    StartRun {
        /// Unique run identifier for the background task invocation.
        run_id: String,
        /// Session that owns the run.
        session_id: String,
        /// User input that started the run.
        input: String,
        /// Optional structured instructions appended to the request.
        instructions: Option<serde_json::Value>,
        /// Explicit model requested for the run, if any.
        model_id: Option<String>,
        /// Optional stored prompt collection selected for the run.
        prompt_collection_id: Option<String>,
        /// Typed reasoning controls for the run.
        reasoning: ReasoningSettings,
    },
    /// Drain queued runs when a compatible node becomes available.
    DrainQueue,
}

/// Handle used by request handlers to submit work to the run task.
#[derive(Clone)]
pub struct RunHandle {
    cmd_tx: mpsc::Sender<RunCommand>,
}

impl RunHandle {
    /// Spawn the background run task and return a handle to it.
    pub fn spawn(db: SqlitePool, state: SharedState, event_tx: broadcast::Sender<Frame>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        tokio::spawn(run_task(cmd_rx, db, state, event_tx));
        Self { cmd_tx }
    }

    /// Submit a run command to the background task.
    pub async fn submit(&self, cmd: RunCommand) -> Result<(), mpsc::error::SendError<RunCommand>> {
        self.cmd_tx.send(cmd).await
    }
}

/// Process run commands until the sender side is dropped.
async fn run_task(
    mut cmd_rx: mpsc::Receiver<RunCommand>,
    db: SqlitePool,
    state: SharedState,
    event_tx: broadcast::Sender<Frame>,
) {
    tracing::info!("Agent Loop started");
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            RunCommand::StartRun {
                run_id,
                session_id,
                input,
                instructions,
                model_id,
                prompt_collection_id,
                reasoning,
            } => {
                tracing::info!(
                    "Starting run {run_id} (session={session_id}, reasoning={reasoning:?})"
                );
                super::r#loop::start_run(
                    &run_id,
                    &session_id,
                    &input,
                    instructions.as_ref(),
                    &db,
                    &state,
                    &event_tx,
                    model_id.as_deref(),
                    prompt_collection_id.as_deref(),
                    reasoning,
                )
                .await;
            }
            RunCommand::DrainQueue => {
                tracing::info!("Queue drain triggered");
                super::queue::drain_queue(&db, &state, &event_tx).await;
            }
        }
    }
    tracing::info!("Agent Loop shut down");
}
