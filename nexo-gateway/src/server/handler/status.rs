use crate::agent::{AgentCommand, AgentHandle};
use crate::server::state::SharedState;
use nexo_ws_schema::{Frame, HealthResponse, ModelStatusParams, StatusResponse, ToolsCatalogResponse};

use super::base::{ok_or_internal_error, parse_params};

pub(super) async fn handle_health(request_id: &str, state: &SharedState) -> Frame {
    let state = state.read().await;
    ok_or_internal_error(
        request_id,
        HealthResponse {
            status: "ok".into(),
            uptime_secs: state.uptime_secs(),
        },
    )
}

pub(super) async fn handle_status(request_id: &str, state: &SharedState) -> Frame {
    let state = state.read().await;
    ok_or_internal_error(
        request_id,
        StatusResponse {
            connected_users: state.connected_users(),
            connected_nodes: state.connected_nodes(),
            capabilities: state.all_capabilities(),
        },
    )
}

pub(super) async fn handle_tools_catalog(request_id: &str, state: &SharedState) -> Frame {
    let state = state.read().await;
    ok_or_internal_error(
        request_id,
        ToolsCatalogResponse {
            tools: state.all_tool_entries(),
        },
    )
}

pub(super) async fn handle_model_status(
    request_id: &str,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    agent_handle: &AgentHandle,
) -> Frame {
    let status_params: ModelStatusParams =
        match parse_params(request_id, params, "model.status") {
            Ok(p) => p,
            Err(f) => return f,
        };
    let model_became_available = !status_params.loaded_models.is_empty();
    {
        let mut sw = state.write().await;
        sw.set_loaded_models(peer_id, status_params.loaded_models);
        sw.set_available_models(peer_id, status_params.available_models);
    }
    if model_became_available
        && let Err(e) = agent_handle.submit(AgentCommand::DrainQueue).await
    {
        tracing::warn!("Failed to submit DrainQueue after ModelStatus: {e}");
    }

    ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
}
