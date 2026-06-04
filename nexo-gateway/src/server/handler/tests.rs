#![allow(clippy::unwrap_used, clippy::panic)]
use super::*;
use crate::agent::RunHandle;
use crate::server::state::{GatewayState, PeerInfo, SharedState, dummy_sender};
use nexo_core::{
    MetadataMap, ModelCapability, ModelId, ModelModalities, RoleStrategy, SupportedModality,
};
use nexo_ws_schema::{ConnectionRole, EventKind, Frame, Method};
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, timeout};

fn make_state() -> SharedState {
    Arc::new(RwLock::new(GatewayState::new(std::path::PathBuf::from(
        "/tmp",
    ))))
}

fn make_run_handle(state: &SharedState, db: &SqlitePool) -> RunHandle {
    let event_tx = {
        let st = state.try_read().unwrap();
        st.event_tx.clone()
    };
    RunHandle::spawn(db.clone(), state.clone(), event_tx)
}

fn make_loaded_model(
    model_id: &str,
    capabilities: Vec<ModelCapability>,
) -> nexo_core::ModelDescriptor {
    nexo_core::ModelDescriptor {
        id: ModelId::from(model_id),
        display_name: model_id.into(),
        provider: Some("test".into()),
        capabilities,
        modalities: ModelModalities {
            input: vec![SupportedModality::Text],
            output: vec![SupportedModality::Image],
        },
        role_strategy: RoleStrategy::Default,
        context_window_tokens: Some(4096),
        max_output_tokens: Some(1024),
        metadata: MetadataMap::new(),
    }
}

// Helper: dispatch with a real DB pool
async fn dispatch(
    req_id: &str,
    method: &Method,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    run_handle: &RunHandle,
) -> Frame {
    dispatch_method(req_id, method, params, peer_id, state, db, run_handle).await
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_health_returns_ok(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    let resp = dispatch(
        "req-1",
        &Method::Health,
        serde_json::json!({}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let p = payload.unwrap();
        assert_eq!(p["status"], "ok");
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_status_returns_counts(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }
    let resp = dispatch(
        "req-1",
        &Method::Status,
        serde_json::json!({}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let p = payload.unwrap();
        assert_eq!(p["connectedUsers"], 1);
        assert_eq!(p["connectedNodes"], 0);
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_connect_after_handshake_rejected(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    let resp = dispatch(
        "req-1",
        &Method::Connect,
        serde_json::json!({}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, error, .. } = resp {
        assert!(!ok);
        assert_eq!(error.unwrap().code, "invalid_method");
    } else {
        panic!("Expected error response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_tools_register_from_node(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "n1".into(),
                client_id: "node".into(),
                role: ConnectionRole::Node,
                scopes: vec![],
                capabilities: vec!["echo".into()],
                commands: vec!["echo.run".into()],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let params = serde_json::json!({
        "tools": [{
            "name": "echo.run",
            "description": "Echo input",
            "parameters": {"type": "object"}
        }]
    });

    let resp = dispatch(
        "req-1",
        &Method::ToolsRegister,
        params,
        "n1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        assert_eq!(payload.unwrap()["registered"], 1);
    } else {
        panic!("Expected response");
    }

    // Verify tool is in catalog
    let catalog_resp = dispatch(
        "req-2",
        &Method::ToolsCatalog,
        serde_json::json!({}),
        "n1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = catalog_resp {
        assert!(ok);
        let tools = &payload.unwrap()["tools"];
        assert_eq!(tools.as_array().unwrap().len(), 1);
        assert_eq!(tools[0]["name"], "echo.run");
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_tools_register_from_user_rejected(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "u1".into(),
                client_id: "cli".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let params = serde_json::json!({"tools": []});
    let resp = dispatch(
        "req-1",
        &Method::ToolsRegister,
        params,
        "u1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, error, .. } = resp {
        assert!(!ok);
        assert_eq!(error.unwrap().code, "forbidden");
    } else {
        panic!("Expected error response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_tools_execute_tool_not_found(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    let params = serde_json::json!({
        "tool": "nonexistent",
        "args": {},
        "idempotencyKey": "k1"
    });
    let resp = dispatch(
        "req-1",
        &Method::ToolsExecute,
        params,
        "u1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, error, .. } = resp {
        assert!(!ok);
        assert_eq!(error.unwrap().code, "tool_not_found");
    } else {
        panic!("Expected error response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_send_routes_message_to_target_user(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    let (target_tx, mut target_rx) = mpsc::channel(1);

    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "sender-peer".into(),
                client_id: "user-a".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
        s.add_peer(
            PeerInfo {
                id: "target-peer".into(),
                client_id: "user-b".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            target_tx,
        );
    }

    let resp = dispatch(
        "req-1",
        &Method::Send,
        serde_json::json!({
            "target": "user-b",
            "payload": {"text": "hello"},
            "idempotencyKey": "k1"
        }),
        "sender-peer",
        &state,
        &pool,
        &ah,
    )
    .await;

    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        assert_eq!(payload.unwrap()["delivered"], true);
    } else {
        panic!("Expected response");
    }

    match target_rx.recv().await {
        Some(Frame::Event { event, payload, .. }) => {
            assert_eq!(event, EventKind::Message);
            assert_eq!(payload["from"], "user-a");
            assert_eq!(payload["target"], "user-b");
            assert_eq!(payload["payload"]["text"], "hello");
            assert!(payload["messageId"].as_str().is_some());
        }
        other => panic!("Expected message event, got {other:?}"),
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_image_generate_queues_until_capable_node_is_ready(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    let (node_tx, mut node_rx) = mpsc::channel(1);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "n1".into(),
                client_id: "node-1".into(),
                role: ConnectionRole::Node,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            node_tx,
        );
    }

    let mut event_rx = {
        let s = state.read().await;
        s.event_tx.subscribe()
    };

    let state_for_dispatch = state.clone();
    let pool_for_dispatch = pool.clone();
    let ah_for_dispatch = ah.clone();
    let response_task = tokio::spawn(async move {
        dispatch(
            "req-img",
            &Method::ImageGenerate,
            serde_json::json!({
                "prompt": "a smiley face",
                "idempotencyKey": "idem-img-1",
                "sessionId": "sess-queue-1"
            }),
            "p1",
            &state_for_dispatch,
            &pool_for_dispatch,
            &ah_for_dispatch,
        )
        .await
    });

    let queued_event = timeout(Duration::from_secs(2), event_rx.recv())
        .await
        .expect("timed out waiting for queued event")
        .expect("failed to receive queued event");
    match queued_event {
        Frame::Event { event, payload, .. } => {
            assert_eq!(event, EventKind::Message);
            assert_eq!(payload["payload"]["kind"], "generation.queued");
            assert_eq!(payload["payload"]["method"], "image.generate");
            assert_eq!(payload["payload"]["sessionId"], "sess-queue-1");
            assert_eq!(payload["payload"]["queuedCount"], 1);
        }
        other => panic!("Expected queued message event, got {other:?}"),
    }

    {
        let mut s = state.write().await;
        s.set_loaded_models(
            "n1",
            vec![make_loaded_model(
                "flux.2-klein-9b",
                vec![ModelCapability::ImageGeneration],
            )],
        );
    }

    let forwarded = timeout(Duration::from_secs(2), node_rx.recv())
        .await
        .expect("timed out waiting for forwarded request")
        .expect("node channel closed");
    let forwarded_id = match forwarded {
        Frame::Request { id, method, .. } => {
            assert_eq!(method, Method::ImageGenerate);
            id
        }
        other => panic!("Expected forwarded image.generate request, got {other:?}"),
    };

    let pending_tx = {
        let mut s = state.write().await;
        s.pending_requests
            .remove(&forwarded_id)
            .expect("expected pending request sender")
    };
    pending_tx
        .send(Frame::Response {
            id: forwarded_id,
            ok: true,
            payload: Some(serde_json::json!({
                "images": [{"index": 0, "imageData": "ZmFrZQ==", "mediaType": "image/png"}],
                "inferenceTimeMs": 10
            })),
            error: None,
        })
        .expect("failed to send simulated node response");

    let response = timeout(Duration::from_secs(2), response_task)
        .await
        .expect("timed out waiting for gateway response")
        .expect("dispatch task failed to join");
    if let Frame::Response { ok, payload, .. } = response {
        assert!(ok);
        assert_eq!(payload.unwrap()["images"][0]["index"], 0);
    } else {
        panic!("Expected response");
    }

    let queued_count = {
        let s = state.read().await;
        s.queued_generation_by_session
            .get("sess-queue-1")
            .copied()
            .unwrap_or(0)
    };
    assert_eq!(queued_count, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_send_to_unknown_target_reports_not_delivered(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "sender-peer".into(),
                client_id: "user-a".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let resp = dispatch(
        "req-1",
        &Method::Send,
        serde_json::json!({
            "target": "user-b",
            "payload": {"text": "hello"},
            "idempotencyKey": "k1"
        }),
        "sender-peer",
        &state,
        &pool,
        &ah,
    )
    .await;

    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        assert_eq!(payload.unwrap()["delivered"], false);
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_send_from_node_rejected(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "node-peer".into(),
                client_id: "node-a".into(),
                role: ConnectionRole::Node,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let resp = dispatch(
        "req-1",
        &Method::Send,
        serde_json::json!({
            "target": "user-b",
            "payload": {"text": "hello"},
            "idempotencyKey": "k1"
        }),
        "node-peer",
        &state,
        &pool,
        &ah,
    )
    .await;

    if let Frame::Response { ok, error, .. } = resp {
        assert!(!ok);
        assert_eq!(error.unwrap().code, "forbidden");
    } else {
        panic!("Expected error response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_session_create_returns_id(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    // Set up user FK
    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    // Add peer so user_id_for_peer resolves
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let params = serde_json::json!({"name": "test session"});
    let resp = dispatch(
        "req-1",
        &Method::SessionCreate,
        params,
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let p = payload.unwrap();
        assert!(p["sessionId"].as_str().is_some());
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_session_list_empty(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let resp = dispatch(
        "req-1",
        &Method::SessionList,
        serde_json::json!({}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let sessions = payload.unwrap()["sessions"].as_array().unwrap().clone();
        assert!(sessions.is_empty());
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_run_start_returns_accepted_with_session(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();

    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: ConnectionRole::User,
                scopes: vec![],
                capabilities: vec![],
                commands: vec![],
                device_id: None,
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
    }

    let params = serde_json::json!({
        "input": "hello",
        "idempotencyKey": "k1"
    });
    let resp = dispatch("req-1", &Method::RunStart, params, "p1", &state, &pool, &ah).await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let p = payload.unwrap();
        assert_eq!(p["status"], "accepted");
        assert!(p["runId"].as_str().is_some());
        assert!(p["sessionId"].as_str().is_some());
    } else {
        panic!("Expected response");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_run_stop_marks_run_cancelled(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('sess-1', 'cli')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO runs (id, session_id, idempotency_key, status) VALUES ('run-1', 'sess-1', 'idem-1', 'accepted')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let resp = dispatch(
        "req-1",
        &Method::RunStop,
        serde_json::json!({"runId": "run-1"}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        assert_eq!(payload.unwrap()["stopped"], true);
    } else {
        panic!("Expected response");
    }

    let (status, finished_at): (String, Option<String>) =
        sqlx::query_as("SELECT status, finished_at FROM runs WHERE id = 'run-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(status, "cancelled");
    assert!(finished_at.is_some());
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_run_instructions_append_persists_message(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    sqlx::query("INSERT INTO devices (id, role) VALUES ('dev-1', 'user')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users (id, device_id) VALUES ('cli', 'dev-1')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO sessions (id, user_id) VALUES ('sess-1', 'cli')")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO runs (id, session_id, idempotency_key, status) VALUES ('run-1', 'sess-1', 'idem-1', 'accepted')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let resp = dispatch(
        "req-1",
        &Method::RunInstructionsAppend,
        serde_json::json!({
            "runId": "run-1",
            "instructions": {"notes": ["agent_loop.md"]}
        }),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        let payload = payload.unwrap();
        assert_eq!(payload["queued"], true);
        assert!(payload["messageId"].as_str().is_some());
    } else {
        panic!("Expected response");
    }

    let (role, content): (String, String) = sqlx::query_as(
        "SELECT role, content FROM conversation_entries WHERE run_id = 'run-1' ORDER BY created_at DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(role, "system");
    assert!(content.contains("agent_loop.md"));
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_cron_create_and_list(pool: SqlitePool) {
    let state = make_state();
    let ah = make_run_handle(&state, &pool);

    let params = serde_json::json!({
        "name": "test job",
        "schedule": "0 * * * *",
        "input": "hello"
    });
    let resp = dispatch(
        "req-1",
        &Method::CronCreate,
        params,
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = resp {
        assert!(ok);
        assert!(payload.unwrap()["jobId"].as_str().is_some());
    } else {
        panic!("Expected response");
    }

    let list_resp = dispatch(
        "req-2",
        &Method::CronList,
        serde_json::json!({}),
        "p1",
        &state,
        &pool,
        &ah,
    )
    .await;
    if let Frame::Response { ok, payload, .. } = list_resp {
        assert!(ok);
        let jobs = payload.unwrap()["jobs"].as_array().unwrap().clone();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0]["name"], "test job");
    } else {
        panic!("Expected response");
    }
}
