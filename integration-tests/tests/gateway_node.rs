#![allow(clippy::unwrap_used)]

use nexo_gateway::testing::start_test_gateway;
use nexo_spec::model::{LoadedModelInfo, ModelCategory};
use nexo_ws_client::{
    NexoConnection, default_node_connect_params, default_user_connect_params, perform_handshake,
};
use nexo_ws_schema::{
    EventKind, Frame, MessagePayload, Method, ModelStatusParams, Platform, SendParams,
    StatusResponse, ToolSpecEntry, ToolsRegisterParams,
};

// ── Helpers ────────────────────────────────────────────────────────────────

async fn connect_node(
    addr: std::net::SocketAddr,
    capabilities: Vec<String>,
    commands: Vec<String>,
    models: Vec<String>,
) -> NexoConnection {
    let url = format!("ws://{addr}");
    let mut conn = NexoConnection::connect(&url, nexo_ws_schema::AUTH_TOKEN)
        .await
        .expect("node connect failed");

    let params = default_node_connect_params(
        "test-node",
        "0.0.0-test",
        Platform::current(),
        "test-device",
        capabilities,
        commands,
        models,
    );
    perform_handshake(&mut conn, params)
        .await
        .expect("node handshake failed");

    conn
}

async fn connect_user(addr: std::net::SocketAddr) -> NexoConnection {
    connect_user_with_id(addr, "test-user", "test-user-device").await
}

async fn connect_user_with_id(
    addr: std::net::SocketAddr,
    client_id: &str,
    device_id: &str,
) -> NexoConnection {
    let url = format!("ws://{addr}");
    let mut conn = NexoConnection::connect(&url, nexo_ws_schema::AUTH_TOKEN)
        .await
        .expect("user connect failed");

    let params = default_user_connect_params(
        client_id,
        "0.0.0-test",
        Platform::current(),
        device_id,
    );
    perform_handshake(&mut conn, params)
        .await
        .expect("user handshake failed");

    conn
}

async fn register_tools(conn: &mut NexoConnection, tools: Vec<ToolSpecEntry>) -> u32 {
    let frame = Frame::request(Method::ToolsRegister, &ToolsRegisterParams { tools })
        .expect("build register frame");
    conn.send_frame(&frame).await.expect("send register");

    loop {
        let resp = conn
            .recv_frame()
            .await
            .expect("recv register response")
            .expect("connection closed");
        match resp {
            Frame::Response {
                ok: true, payload, ..
            } => {
                return payload
                    .and_then(|p| p.get("registered").and_then(|v| v.as_u64()))
                    .unwrap_or(0) as u32;
            }
            Frame::Event { .. } => continue,
            other => panic!("unexpected frame during registration: {other:?}"),
        }
    }
}

async fn push_model_status(
    conn: &mut NexoConnection,
    loaded: Vec<LoadedModelInfo>,
    available: Vec<String>,
) {
    let status = ModelStatusParams {
        loaded_models: loaded,
        available_models: available,
    };
    let frame = Frame::request(Method::ModelStatus, &status).expect("build model status frame");
    conn.send_frame(&frame).await.expect("send model status");

    // Wait for acknowledgment (skip events)
    loop {
        let resp = conn
            .recv_frame()
            .await
            .expect("recv model status response")
            .expect("connection closed");
        match resp {
            Frame::Response { ok: true, .. } => break,
            Frame::Event { .. } => continue,
            other => panic!("unexpected frame during model status: {other:?}"),
        }
    }
}

async fn request_status(conn: &mut NexoConnection) -> StatusResponse {
    let frame = Frame::request(Method::Status, &serde_json::json!({})).expect("build status frame");
    conn.send_frame(&frame).await.expect("send status");

    loop {
        let resp = conn
            .recv_frame()
            .await
            .expect("recv status response")
            .expect("connection closed");
        match resp {
            Frame::Response {
                ok: true, payload, ..
            } => {
                let payload = payload.expect("status payload missing");
                return serde_json::from_value(payload).expect("parse StatusResponse");
            }
            Frame::Event { .. } => continue,
            other => panic!("unexpected frame during status request: {other:?}"),
        }
    }
}

async fn send_client_message(
    conn: &mut NexoConnection,
    target: &str,
    payload: serde_json::Value,
) -> bool {
    let frame = Frame::request(
        Method::Send,
        &SendParams {
            target: target.to_string(),
            payload,
            idempotency_key: Frame::new_id(),
        },
    )
    .expect("build send frame");
    conn.send_frame(&frame).await.expect("send client message");

    loop {
        let resp = conn
            .recv_frame()
            .await
            .expect("recv send response")
            .expect("connection closed");
        match resp {
            Frame::Response {
                ok: true, payload, ..
            } => {
                return payload
                    .and_then(|value| value.get("delivered").and_then(|flag| flag.as_bool()))
                    .unwrap_or(false);
            }
            Frame::Event { .. } => continue,
            other => panic!("unexpected frame during send request: {other:?}"),
        }
    }
}

async fn recv_message_event(conn: &mut NexoConnection) -> MessagePayload {
    loop {
        let frame = conn
            .recv_frame()
            .await
            .expect("recv message event")
            .expect("connection closed");
        match frame {
            Frame::Event {
                event: EventKind::Message,
                payload,
                ..
            } => return serde_json::from_value(payload).expect("parse MessagePayload"),
            Frame::Event { .. } => continue,
            other => panic!("unexpected frame while waiting for message event: {other:?}"),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn node_connects_with_tools_and_models() {
    let gw = start_test_gateway().await;

    // Connect node with tools and models
    let mut node = connect_node(
        gw.addr,
        vec!["echo".into(), "ping".into()],
        vec!["echo.run".into(), "ping".into()],
        vec!["gemma-4-e4b-it".into()],
    )
    .await;

    // Register tools
    let tools = vec![
        ToolSpecEntry {
            name: "echo.run".into(),
            description: "Echo input".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
        ToolSpecEntry {
            name: "ping".into(),
            description: "Returns pong".into(),
            parameters: serde_json::json!({"type": "object"}),
        },
    ];
    let registered = register_tools(&mut node, tools).await;
    assert_eq!(registered, 2);

    // Push model status (no models loaded, but available on disk)
    push_model_status(&mut node, vec![], vec!["gemma-4-e4b-it".into()]).await;

    // Connect user and verify gateway state
    let mut user = connect_user(gw.addr).await;
    let status = request_status(&mut user).await;

    assert_eq!(status.connected_nodes, 1);
    assert_eq!(status.connected_users, 1);
    assert!(status.capabilities.contains(&"echo".to_string()));
    assert!(status.capabilities.contains(&"ping".to_string()));
}

#[tokio::test]
async fn node_connects_with_tools_only() {
    let gw = start_test_gateway().await;

    let mut node = connect_node(
        gw.addr,
        vec!["echo".into()],
        vec!["echo.run".into()],
        vec![],
    )
    .await;

    let tools = vec![ToolSpecEntry {
        name: "echo.run".into(),
        description: "Echo".into(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let registered = register_tools(&mut node, tools).await;
    assert_eq!(registered, 1);

    // No models to report
    push_model_status(&mut node, vec![], vec![]).await;

    let mut user = connect_user(gw.addr).await;
    let status = request_status(&mut user).await;
    assert_eq!(status.connected_nodes, 1);
    assert!(status.capabilities.contains(&"echo".to_string()));

    // Gateway should have no LLM available
    let state = gw.state.read().await;
    assert!(!state.has_llm_peer());
}

#[tokio::test]
async fn node_connects_with_models_only() {
    let gw = start_test_gateway().await;

    let mut node = connect_node(gw.addr, vec![], vec![], vec!["gemma-4-e4b-it".into()]).await;

    // No tools to register, push model status
    push_model_status(&mut node, vec![], vec!["gemma-4-e4b-it".into()]).await;

    let state = gw.state.read().await;
    assert_eq!(state.connected_nodes(), 1);
    // Model is available but not loaded
    assert!(!state.has_llm_peer());
    // Available models tracked
    let available = state.available_models.values().next().unwrap();
    assert!(available.contains(&"gemma-4-e4b-it".to_string()));
}

#[tokio::test]
async fn node_connects_with_nothing() {
    let gw = start_test_gateway().await;

    let _node = connect_node(gw.addr, vec![], vec![], vec![]).await;

    // Give the gateway a moment to register the peer
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let state = gw.state.read().await;
    assert_eq!(state.connected_nodes(), 1);
    assert!(!state.has_llm_peer());
    assert!(state.all_capabilities().is_empty());
}

#[tokio::test]
async fn model_status_advertisement_makes_llm_available() {
    let gw = start_test_gateway().await;

    let mut node = connect_node(gw.addr, vec![], vec![], vec!["gemma-4-e4b-it".into()]).await;

    // Initially push no loaded models
    push_model_status(&mut node, vec![], vec!["gemma-4-e4b-it".into()]).await;

    {
        let state = gw.state.read().await;
        assert!(!state.has_llm_peer());
    }

    // Now push status with a loaded model that has Chat + Tool categories
    push_model_status(
        &mut node,
        vec![LoadedModelInfo {
            model_id: "gemma-4-e4b-it".into(),
            categories: vec![
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image,
            ],
        }],
        vec!["gemma-4-e4b-it".into()],
    )
    .await;

    let state = gw.state.read().await;
    assert!(state.has_llm_peer());
    let loaded = state.loaded_models.values().next().unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].model_id, "gemma-4-e4b-it");
    assert!(loaded[0].categories.contains(&ModelCategory::Chat));
    assert!(loaded[0].categories.contains(&ModelCategory::Tool));
    assert!(loaded[0].categories.contains(&ModelCategory::Image));
}

#[tokio::test]
async fn multiple_nodes_tracked_independently() {
    let gw = start_test_gateway().await;

    // Connect node 1 with tools
    let mut node1 = connect_node(
        gw.addr,
        vec!["echo".into()],
        vec!["echo.run".into()],
        vec![],
    )
    .await;
    register_tools(
        &mut node1,
        vec![ToolSpecEntry {
            name: "echo.run".into(),
            description: "Echo".into(),
            parameters: serde_json::json!({"type": "object"}),
        }],
    )
    .await;
    push_model_status(&mut node1, vec![], vec![]).await;

    // Connect node 2 with models
    let mut node2 = connect_node(gw.addr, vec![], vec![], vec!["gemma-4-e4b-it".into()]).await;
    push_model_status(
        &mut node2,
        vec![LoadedModelInfo {
            model_id: "gemma-4-e4b-it".into(),
            categories: vec![ModelCategory::Chat, ModelCategory::Tool],
        }],
        vec!["gemma-4-e4b-it".into()],
    )
    .await;

    let mut user = connect_user(gw.addr).await;
    let status = request_status(&mut user).await;
    assert_eq!(status.connected_nodes, 2);
    assert!(status.capabilities.contains(&"echo".to_string()));

    let state = gw.state.read().await;
    assert!(state.has_llm_peer());
}

#[tokio::test]
async fn user_can_send_message_to_other_user_via_gateway() {
    let gw = start_test_gateway().await;

    let mut alice = connect_user_with_id(gw.addr, "alice", "alice-device").await;
    let mut bob = connect_user_with_id(gw.addr, "bob", "bob-device").await;

    let delivered =
        send_client_message(&mut alice, "bob", serde_json::json!({"text": "hello bob"}))
            .await;

    assert!(delivered);

    let message = recv_message_event(&mut bob).await;
    assert_eq!(message.from, "alice");
    assert_eq!(message.target, "bob");
    assert_eq!(message.payload["text"], "hello bob");
    assert!(!message.message_id.is_empty());
}
