//! Application bootstrap that wires storage, background tasks, and the WebSocket server.

use crate::agent::RunHandle;
use crate::config::GatewayConfig;
use crate::memory::git::GitStorage;
use crate::server::state::{GatewayState, SharedState};
use crate::tools::GatewayToolExecutor;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

const CRON_NOTES_SUMMARY: &str = "notes-summary";

/// Start the gateway runtime using the supplied configuration.
pub async fn run(config: &GatewayConfig) -> cli_helpers::Result {
    info!(
        "Gateway config: host={}, port={}, tick_interval={}ms, db={}, storage={}",
        config.host, config.port, config.tick_interval_ms, config.db_path, config.storage_root,
    );

    let db_path = cli_helpers::resolve_path_str(&config.db_path)?;
    let db = crate::memory::persistent::connect(&db_path).await?;
    info!("Database connected: {}", db_path.display());

    let storage_root = cli_helpers::resolve_path_str(&config.storage_root)?;
    info!("Storage root: {}", storage_root.display());

    let git_storage = open_git_storage(config).await;
    let gateway_tools = build_gateway_tools(git_storage.as_ref());

    info!(
        "Registered {} gateway tool(s)",
        gateway_tools.tool_entries().len(),
    );

    let mut gateway_state = GatewayState::new(storage_root);
    gateway_state.gateway_tools = gateway_tools;
    gateway_state.git_storage = git_storage;
    let event_tx = gateway_state.event_tx.clone();
    let state: SharedState = Arc::new(RwLock::new(gateway_state));

    let run_handle = RunHandle::spawn(db.clone(), state.clone(), event_tx.clone());

    seed_default_cron_jobs(&db);
    spawn_background_tasks(config, &db, &event_tx, &run_handle);

    let addr = config.bind_addr();
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|error| cli_helpers::Error::Network(format!("Failed to bind {addr}: {error}")))?;

    info!("NEXO Gateway listening on ws://{addr}");

    let auth_token: Arc<str> = config.auth_token.as_str().into();

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .map_err(|error| cli_helpers::Error::Network(format!("Accept failed: {error}")))?;

        debug!("New TCP connection from {peer_addr}");

        let state = state.clone();
        let db = db.clone();
        let auth_token = auth_token.clone();
        let run_handle = run_handle.clone();

        tokio::spawn(async move {
            #[allow(clippy::result_large_err)]
            let callback =
                |req: &http::Request<()>,
                 resp: http::Response<()>|
                 -> Result<http::Response<()>, http::Response<Option<String>>> {
                    if crate::server::auth::validate_auth(req.headers(), &auth_token) {
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
                Ok(stream) => stream,
                Err(error) => {
                    warn!("WS handshake failed from {peer_addr}: {error}");
                    return;
                }
            };

            let event_rx = state.read().await.event_tx.subscribe();
            crate::server::handler::handle_connection(ws_stream, state, db, event_rx, run_handle)
                .await;
        });
    }
}

/// Open the optional git-backed storage used by gateway-local tools and prefill data.
async fn open_git_storage(config: &GatewayConfig) -> Option<Arc<GitStorage>> {
    match cli_helpers::resolve_path_str(&config.nexo_storage_path) {
        Ok(path) => match GitStorage::open(&path) {
            Ok(storage) => {
                let storage = Arc::new(storage);
                let storage_for_pull = storage.clone();
                tokio::task::spawn_blocking(move || {
                    info!(
                        "Performing git pull on startup for storage at {}",
                        path.display()
                    );
                    if let Err(error) = storage_for_pull.pull() {
                        warn!("Git pull on startup failed: {error}");
                    }
                });
                Some(storage)
            }
            Err(error) => {
                warn!(
                    "Could not open nexo-storage at {}: {error}",
                    config.nexo_storage_path
                );
                None
            }
        },
        Err(error) => {
            warn!("Could not resolve nexo_storage_path: {error}");
            None
        }
    }
}

/// Build the registry of tools that run directly inside the gateway process.
fn build_gateway_tools(git_storage: Option<&Arc<GitStorage>>) -> GatewayToolExecutor {
    let mut gateway_tools = GatewayToolExecutor::new();
    if let Some(storage) = git_storage {
        for tool in nexo_notes::tools::all_tools(storage.clone()) {
            gateway_tools.register(tool);
        }
    }

    for tool in nexo_io::tools::all_tools() {
        gateway_tools.register(tool);
    }

    gateway_tools
}

/// Seed any default cron jobs that should exist for a fresh gateway instance.
fn seed_default_cron_jobs(db: &sqlx::SqlitePool) {
    let seed_db = db.clone();
    tokio::spawn(async move {
        let jobs = crate::agent::cron::list_jobs(&seed_db)
            .await
            .unwrap_or_default();
        if !jobs.iter().any(|job| job.name == CRON_NOTES_SUMMARY) {
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

/// Spawn the long-lived background tasks that support the running gateway.
fn spawn_background_tasks(
    config: &GatewayConfig,
    db: &sqlx::SqlitePool,
    event_tx: &tokio::sync::broadcast::Sender<nexo_ws_schema::Frame>,
    run_handle: &RunHandle,
) {
    let cron_handle = run_handle.clone();
    let cron_db = db.clone();
    let cron_event_tx = event_tx.clone();
    tokio::spawn(async move {
        crate::agent::cron::run_scheduler(cron_db, cron_handle, cron_event_tx).await;
    });

    let tick_tx = event_tx.clone();
    let tick_interval = config.tick_interval_ms;
    tokio::spawn(async move {
        crate::server::ticker::run_ticker(tick_tx, tick_interval).await;
    });
}
