pub mod auth;
pub mod handler;
pub mod state;
pub mod ticker;

use crate::agent::AgentHandle;
use crate::agent::gateway_tools::GatewayToolExecutor;
use crate::config::GatewayConfig;
use crate::memory::git::GitStorage;
use tracing::{debug, info, warn};

const CRON_NOTES_SUMMARY: &str = "notes-summary";
use state::{GatewayState, SharedState};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

/// Start the gateway WebSocket server.
pub async fn run(config: &GatewayConfig) -> utl_helpers::Result {
    info!(
        "Gateway config: host={}, port={}, tick_interval={}ms, db={}, storage={}",
        config.host, config.port, config.tick_interval_ms, config.db_path, config.storage_root,
    );

    let db_path = utl_helpers::resolve_path_str(&config.db_path)?;
    let db = crate::memory::persistent::connect(&db_path).await?;
    info!("Database connected: {}", db_path.display());

    let storage_root = utl_helpers::resolve_path_str(&config.storage_root)?;
    info!("Storage root: {}", storage_root.display());

    // Open git-backed storage (optional — allows gateway to run without persistent storage)
    let git_storage = match utl_helpers::resolve_path_str(&config.nexo_storage_path) {
        Ok(path) => match GitStorage::open(&path) {
            Ok(gs) => {
                let gs = Arc::new(gs);
                let gs_pull = gs.clone();
                tokio::task::spawn_blocking(move || {
                    info!(
                        "Performing git pull on startup for storage at {}",
                        path.display()
                    );
                    if let Err(e) = gs_pull.pull() {
                        warn!("Git pull on startup failed: {e}");
                    }
                });
                Some(gs)
            }
            Err(e) => {
                warn!(
                    "Could not open nexo-storage at {}: {e}",
                    config.nexo_storage_path
                );
                None
            }
        },
        Err(e) => {
            warn!("Could not resolve nexo_storage_path: {e}");
            None
        }
    };

    // Build gateway-native tools
    let mut gateway_tools = GatewayToolExecutor::new();
    if let Some(ref gs) = git_storage {
        for tool in nexo_notes::tools::all_tools(gs.clone()) {
            gateway_tools.register(tool);
        }
    }

    // Register io tools (unconditional — no storage dependency)
    for tool in nexo_io::tools::all_tools() {
        gateway_tools.register(tool);
    }

    info!(
        "Registered {} gateway tool(s)",
        gateway_tools.tool_entries().len(),
    );

    let mut gateway_state = GatewayState::new(storage_root);
    gateway_state.gateway_tools = gateway_tools;
    gateway_state.git_storage = git_storage;
    let event_tx = gateway_state.event_tx.clone();
    let state: SharedState = Arc::new(RwLock::new(gateway_state));

    // Spawn the agent brain
    let agent_handle = AgentHandle::spawn(db.clone(), state.clone(), event_tx.clone());

    // Seed default cron jobs (idempotent)
    {
        let seed_db = db.clone();
        tokio::spawn(async move {
            let jobs = crate::agent::cron::list_jobs(&seed_db)
                .await
                .unwrap_or_default();
            if !jobs.iter().any(|j| j.name == CRON_NOTES_SUMMARY) {
                let _ = crate::agent::cron::create_job(
                    &seed_db,
                    CRON_NOTES_SUMMARY,
                    "0 */6 * * *",
                    "Read all notes using notes.list and notes.read. \
                    Write an organized summary using notes.update_summary.",
                    None,
                )
                .await;
            }
        });
    }

    // Spawn the cron scheduler
    let cron_handle = agent_handle.clone();
    let cron_db = db.clone();
    let cron_event_tx = event_tx.clone();
    tokio::spawn(async move {
        crate::agent::cron::run_scheduler(cron_db, cron_handle, cron_event_tx).await;
    });

    // Spawn the tick event broadcaster
    let tick_tx = event_tx.clone();
    let tick_interval = config.tick_interval_ms;
    tokio::spawn(async move {
        ticker::run_ticker(tick_tx, tick_interval).await;
    });

    let addr = config.bind_addr();
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Failed to bind {addr}: {e}")))?;

    info!("NEXO Gateway listening on ws://{addr}");

    let auth_token: Arc<str> = config.auth_token.as_str().into();

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .map_err(|e| utl_helpers::Error::Network(format!("Accept failed: {e}")))?;

        debug!("New TCP connection from {peer_addr}");

        let state = state.clone();
        let db = db.clone();
        let auth_token = auth_token.clone();
        let agent_handle = agent_handle.clone();

        tokio::spawn(async move {
            // Perform WebSocket upgrade with auth check
            #[allow(clippy::result_large_err)]
            let callback =
                |req: &http::Request<()>,
                 resp: http::Response<()>|
                 -> Result<http::Response<()>, http::Response<Option<String>>> {
                    if auth::validate_auth(req.headers(), &auth_token) {
                        Ok(resp)
                    } else {
                        warn!("Auth failed from {peer_addr}");
                        Err(http::Response::builder()
                            .status(401)
                            .body(Some("Unauthorized".into()))
                            .unwrap_or_default())
                    }
                };

            let ws_stream = match tokio_tungstenite::accept_hdr_async(stream, callback).await {
                Ok(ws) => ws,
                Err(e) => {
                    warn!("WS handshake failed from {peer_addr}: {e}");
                    return;
                }
            };

            // Subscribe to events only after successful WS upgrade
            let event_rx = state.read().await.event_tx.subscribe();
            handler::handle_connection(ws_stream, state, db, event_rx, agent_handle).await;
        });
    }
}
