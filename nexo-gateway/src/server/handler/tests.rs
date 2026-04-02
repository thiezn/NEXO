#![allow(clippy::unwrap_used, clippy::panic)]
use super::*;
use crate::agent::{AgentHandle};
use crate::server::state::{GatewayState, PeerInfo, dummy_sender};
use nexo_ws_schema::{Frame, Method, Role};
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::sync::RwLock;

fn make_state() -> SharedState {
    Arc::new(RwLock::new(GatewayState::new(std::path::PathBuf::from(
        "/tmp",
    ))))
}

fn make_agent_handle(state: &SharedState, db: &SqlitePool) -> AgentHandle {
    let event_tx = {
        let st = state.try_read().unwrap();
        st.event_tx.clone()
    };
    AgentHandle::spawn(db.clone(), state.clone(), event_tx)
}

// Helper: dispatch with a real DB pool
async fn dispatch(
    req_id: &str,
    method: &Method,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    agent_handle: &AgentHandle,
) -> Frame {
    dispatch_method(req_id, method, params, peer_id, state, db, agent_handle).await
}

#[sqlx::test(migrations = "./migrations")]
async fn dispatch_health_returns_ok(pool: SqlitePool) {
    let state = make_state();
    let ah = make_agent_handle(&state, &pool);
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
    let ah = make_agent_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: Role::User,
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
    let ah = make_agent_handle(&state, &pool);
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
    let ah = make_agent_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "n1".into(),
                client_id: "node".into(),
                role: Role::Node,
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
    let ah = make_agent_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "u1".into(),
                client_id: "cli".into(),
                role: Role::User,
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
    let ah = make_agent_handle(&state, &pool);
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
async fn dispatch_session_create_returns_id(pool: SqlitePool) {
    let state = make_state();
    let ah = make_agent_handle(&state, &pool);

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
                role: Role::User,
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
    let ah = make_agent_handle(&state, &pool);
    {
        let mut s = state.write().await;
        s.add_peer(
            PeerInfo {
                id: "p1".into(),
                client_id: "cli".into(),
                role: Role::User,
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
async fn dispatch_agent_returns_accepted_with_session(pool: SqlitePool) {
    let state = make_state();
    let ah = make_agent_handle(&state, &pool);

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
                role: Role::User,
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
        "prompt": "hello",
        "idempotencyKey": "k1"
    });
    let resp = dispatch("req-1", &Method::Agent, params, "p1", &state, &pool, &ah).await;
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
async fn dispatch_cron_create_and_list(pool: SqlitePool) {
    let state = make_state();
    let ah = make_agent_handle(&state, &pool);

    let params = serde_json::json!({
        "name": "test job",
        "schedule": "0 * * * *",
        "prompt": "hello"
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
