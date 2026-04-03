//! Test helpers for integration tests. Not intended for production use.

use crate::agent::AgentHandle;
use crate::server::auth;
use crate::server::handler;
use crate::server::state::{GatewayState, SharedState};
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

/// A running test gateway instance.
pub struct TestGateway {
    pub addr: SocketAddr,
    pub state: SharedState,
    pub db: SqlitePool,
}

/// Create an in-memory SQLite database with gateway migrations applied.
pub async fn create_test_db() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .connect("sqlite::memory:")
        .await
        .expect("failed to create in-memory SQLite");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("failed to run gateway migrations");
    pool
}

/// Start a minimal test gateway on a random port.
/// Returns a `TestGateway` with the bound address and shared state.
pub async fn start_test_gateway() -> TestGateway {
    let db = create_test_db().await;

    let gateway_state = GatewayState::new(PathBuf::from("/tmp"));
    let event_tx = gateway_state.event_tx.clone();
    let state: SharedState = Arc::new(RwLock::new(gateway_state));

    let agent_handle = AgentHandle::spawn(db.clone(), state.clone(), event_tx.clone());

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind test gateway");
    let addr = listener.local_addr().expect("no local addr");

    let auth_token: Arc<str> = nexo_ws_schema::AUTH_TOKEN.into();
    let accept_state = state.clone();
    let accept_db = db.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };

            let state = accept_state.clone();
            let db = accept_db.clone();
            let auth_token = auth_token.clone();
            let agent_handle = agent_handle.clone();

            tokio::spawn(async move {
                let callback = |req: &http::Request<()>,
                                resp: http::Response<()>|
                 -> Result<http::Response<()>, http::Response<Option<String>>> {
                    if auth::validate_auth(req.headers(), &auth_token) {
                        Ok(resp)
                    } else {
                        Err(http::Response::builder()
                            .status(401)
                            .body(Some("Unauthorized".into()))
                            .unwrap_or_default())
                    }
                };

                let Ok(ws_stream) =
                    tokio_tungstenite::accept_hdr_async(stream, callback).await
                else {
                    return;
                };

                let event_rx = state.read().await.event_tx.subscribe();
                handler::handle_connection(ws_stream, state, db, event_rx, agent_handle).await;
            });
        }
    });

    TestGateway { addr, state, db }
}
