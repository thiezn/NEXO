pub mod auth;
pub mod handler;
pub mod state;
pub mod ticker;

use crate::agent::AgentHandle;
use crate::config::GatewayConfig;
use state::{GatewayState, SharedState};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

/// Start the gateway WebSocket server.
pub async fn run(config: &GatewayConfig) -> utl_helpers::Result {
    let db_path = utl_helpers::resolve_path_str(&config.db_path)?;
    let db = crate::memory::persistent::connect(&db_path).await?;

    let storage_root = utl_helpers::resolve_path_str(&config.storage_root)?;
    let gateway_state = GatewayState::new(storage_root);
    let event_tx = gateway_state.event_tx.clone();
    let state: SharedState = Arc::new(RwLock::new(gateway_state));

    // Spawn the agent brain
    let agent_handle = AgentHandle::spawn(db.clone(), state.clone(), event_tx.clone());

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

    tracing::info!("NEXO Gateway listening on ws://{addr}");

    let auth_token: Arc<str> = config.auth_token.as_str().into();

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .map_err(|e| utl_helpers::Error::Network(format!("Accept failed: {e}")))?;

        tracing::debug!("New TCP connection from {peer_addr}");

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
                        tracing::warn!("Auth failed from {peer_addr}");
                        Err(http::Response::builder()
                            .status(401)
                            .body(Some("Unauthorized".into()))
                            .unwrap_or_default())
                    }
                };

            let ws_stream = match tokio_tungstenite::accept_hdr_async(stream, callback).await {
                Ok(ws) => ws,
                Err(e) => {
                    tracing::warn!("WS handshake failed from {peer_addr}: {e}");
                    return;
                }
            };

            // Subscribe to events only after successful WS upgrade
            let event_rx = state.read().await.event_tx.subscribe();
            handler::handle_connection(ws_stream, state, db, event_rx, agent_handle).await;
        });
    }
}
