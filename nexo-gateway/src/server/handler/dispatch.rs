//! Request dispatch for already-authenticated peers.

use crate::agent::RunHandle;
use crate::server::state::SharedState;
use nexo_ws_schema::{ErrorPayload, Frame, Method};
use sqlx::SqlitePool;

use super::{base, cron, image_analyze, prompt, run, send, session, status, tools};

/// Dispatch a method request from a connected peer.
pub(crate) async fn dispatch_method(
    request_id: &str,
    method: &Method,
    params: serde_json::Value,
    peer_id: &str,
    state: &SharedState,
    db: &SqlitePool,
    run_handle: &RunHandle,
) -> Frame {
    use base::ok_or_internal_error;

    match method {
        Method::Health => status::handle_health(request_id, state).await,
        Method::Status => status::handle_status(request_id, state).await,
        Method::ToolsCatalog => status::handle_tools_catalog(request_id, state).await,
        Method::ModelStatus => {
            status::handle_model_status(request_id, params, peer_id, state, run_handle).await
        }
        Method::ToolsRegister => tools::handle_register(request_id, params, peer_id, state).await,
        Method::ToolsExecute => tools::handle_execute(request_id, params, peer_id, state).await,
        Method::RunStart => {
            run::handle_run_start(request_id, params, peer_id, state, db, run_handle).await
        }
        Method::RunStop => run::handle_run_stop(request_id, params, state, db).await,
        Method::RunInstructionsAppend => {
            run::handle_run_instructions_append(request_id, params, db).await
        }
        Method::SessionCreate => {
            session::handle_create(request_id, params, peer_id, state, db).await
        }
        Method::SessionList => session::handle_list(request_id, peer_id, state, db).await,
        Method::SessionGet => session::handle_get(request_id, params, db).await,
        Method::SessionClear => session::handle_clear(request_id, params, db).await,
        Method::CronCreate => cron::handle_create(request_id, params, db).await,
        Method::CronList => cron::handle_list(request_id, db).await,
        Method::CronDelete => cron::handle_delete(request_id, params, db).await,
        Method::PromptDocumentCreate => {
            prompt::handle_document_create(request_id, params, state).await
        }
        Method::PromptDocumentList => prompt::handle_document_list(request_id, state).await,
        Method::PromptDocumentDelete => {
            prompt::handle_document_delete(request_id, params, state).await
        }
        Method::PromptCollectionCreate => {
            prompt::handle_collection_create(request_id, params, state).await
        }
        Method::PromptCollectionList => prompt::handle_collection_list(request_id, state).await,
        Method::PromptCollectionDelete => {
            prompt::handle_collection_delete(request_id, params, state).await
        }
        Method::RunRound => Frame::error_response(
            request_id,
            ErrorPayload {
                code: "invalid_method".into(),
                message: "This method is only sent by the gateway to nodes".into(),
            },
        ),
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
