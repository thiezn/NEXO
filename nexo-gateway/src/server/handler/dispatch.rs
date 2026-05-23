//! Request dispatch for already-authenticated peers.

use crate::agent::AgentHandle;
use crate::server::state::SharedState;
use nexo_ws_schema::{ErrorPayload, Frame, Method};
use sqlx::SqlitePool;

use super::{agent, base, cron, image_analyze, prefill, send, status, tools};

/// Dispatch a method request from a connected peer.
pub(crate) async fn dispatch_method(
    request_id: &str,
    method: &Method,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    agent_handle: &AgentHandle,
) -> Frame {
    use base::ok_or_internal_error;

    match method {
        Method::Health => status::handle_health(request_id, state).await,
        Method::Status => status::handle_status(request_id, state).await,
        Method::ToolsCatalog => status::handle_tools_catalog(request_id, state).await,
        Method::ModelStatus => {
            status::handle_model_status(request_id, params, peer_id, state, agent_handle).await
        }
        Method::ToolsRegister => tools::handle_register(request_id, params, peer_id, state).await,
        Method::ToolsExecute => tools::handle_execute(request_id, params, peer_id, state).await,
        Method::Agent => {
            agent::handle_agent(request_id, params, peer_id, state, db, agent_handle).await
        }
        Method::AgentStop => agent::handle_agent_stop(request_id, params, state, db).await,
        Method::AgentContextAppend => {
            agent::handle_agent_context_append(request_id, params, db).await
        }
        Method::SessionCreate => {
            agent::handle_session_create(request_id, params, peer_id, state, db).await
        }
        Method::SessionList => agent::handle_session_list(request_id, peer_id, state, db).await,
        Method::SessionGet => agent::handle_session_get(request_id, params, db).await,
        Method::SessionClear => agent::handle_session_clear(request_id, params, db).await,
        Method::CronCreate => cron::handle_create(request_id, params, db).await,
        Method::CronList => cron::handle_list(request_id, db).await,
        Method::CronDelete => cron::handle_delete(request_id, params, db).await,
        Method::PrefillFetch => prefill::handle_fetch_deprecated(request_id),
        Method::PrefillMarkdownCreate => {
            prefill::handle_markdown_create(request_id, params, state).await
        }
        Method::PrefillMarkdownList => prefill::handle_markdown_list(request_id, state).await,
        Method::PrefillMarkdownDelete => {
            prefill::handle_markdown_delete(request_id, params, state).await
        }
        Method::PrefillCollectionCreate => {
            prefill::handle_collection_create(request_id, params, state).await
        }
        Method::PrefillCollectionList => prefill::handle_collection_list(request_id, state).await,
        Method::PrefillCollectionDelete => {
            prefill::handle_collection_delete(request_id, params, state).await
        }
        Method::ImageAnalyze => image_analyze::handle(request_id, params, state).await,
        Method::SystemPresence => {
            ok_or_internal_error(request_id, serde_json::json!({"acknowledged": true}))
        }
        Method::Send => send::handle_send(request_id, params, peer_id, state).await,
        Method::ModelLoad | Method::ModelUnload => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "This method is only sent by the gateway to nodes".into(),
            },
        ),
        Method::Connect => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "Connect can only be the first frame".into(),
            },
        ),
    }
}
