pub mod context;
pub mod cron;
pub mod gateway_tools;
pub mod locks;
pub mod r#loop;
pub mod prefill;
pub mod queue;
pub mod session;
pub mod tool_orchestrator;

use crate::server::state::SharedState;
use nexo_ws_schema::Frame;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc};

/// Commands sent to the agent background task.
pub enum AgentCommand {
    RunAgent {
        run_id: String,
        session_id: String,
        prompt: String,
        context: Option<serde_json::Value>,
        peer_id: String,
        model_id: Option<String>,
        prefill_collection_id: Option<String>,
        thinking: bool,
    },
    /// Drain queued runs when a new LLM node connects.
    DrainQueue,
}

/// Handle through which the handler dispatches agent work.
/// Cheaply cloneable (wraps an mpsc sender).
#[derive(Clone)]
pub struct AgentHandle {
    cmd_tx: mpsc::Sender<AgentCommand>,
}

impl AgentHandle {
    /// Spawn the background agent task and return a handle.
    pub fn spawn(db: SqlitePool, state: SharedState, event_tx: broadcast::Sender<Frame>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        tokio::spawn(agent_task(cmd_rx, db, state, event_tx));
        Self { cmd_tx }
    }

    /// Submit an agent run. The caller has already sent the "accepted" response.
    pub async fn submit(
        &self,
        cmd: AgentCommand,
    ) -> Result<(), mpsc::error::SendError<AgentCommand>> {
        self.cmd_tx.send(cmd).await
    }
}

/// Long-running background task that processes agent commands.
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
                r#loop::start_run(
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
                queue::drain_queue(&db, &state, &event_tx).await;
            }
        }
    }
    tracing::info!("Agent brain shut down");
}
