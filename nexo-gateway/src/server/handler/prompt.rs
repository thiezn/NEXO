//! WebSocket handlers for prompt document and collection management.

use crate::server::state::SharedState;
use nexo_ws_schema::{
    Frame, PromptCollection, PromptCollectionCreateParams, PromptCollectionDeleteParams,
    PromptDocumentCreateParams, PromptDocumentDeleteParams, PromptDocumentEntry,
};

use super::base::{git_blocking, ok_or_internal_error, parse_params};

/// Handle `prompt.document.create` requests.
pub(super) async fn handle_document_create(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PromptDocumentCreateParams =
        match parse_params(request_id, params, "prompt.document.create") {
            Ok(v) => v,
            Err(f) => return f,
        };
    let id = p.id.clone();
    let document = nexo_ws_schema::PromptDocument {
        id: p.id,
        content: p.content,
    };
    match git_blocking(request_id, state, move |git| {
        crate::agent::context::create_prompt_document(&git, &document)
    })
    .await
    {
        Ok(()) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptDocumentCreateResponse { id },
        ),
        Err(frame) => frame,
    }
}

/// Handle `prompt.document.list` requests.
pub(super) async fn handle_document_list(request_id: &str, state: &SharedState) -> Frame {
    match git_blocking(request_id, state, |git| {
        crate::agent::context::list_prompt_documents(&git)
    })
    .await
    {
        Ok(files) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptDocumentListResponse {
                documents: files
                    .into_iter()
                    .map(|id| PromptDocumentEntry { id })
                    .collect(),
            },
        ),
        Err(frame) => frame,
    }
}

/// Handle `prompt.document.delete` requests.
pub(super) async fn handle_document_delete(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PromptDocumentDeleteParams =
        match parse_params(request_id, params, "prompt.document.delete") {
            Ok(v) => v,
            Err(f) => return f,
        };
    match git_blocking(request_id, state, move |git| {
        crate::agent::context::delete_prompt_document(&git, &p.id)
    })
    .await
    {
        Ok(deleted) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptDocumentDeleteResponse { deleted },
        ),
        Err(frame) => frame,
    }
}

/// Handle `prompt.collection.create` requests.
pub(super) async fn handle_collection_create(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PromptCollectionCreateParams =
        match parse_params(request_id, params, "prompt.collection.create") {
            Ok(v) => v,
            Err(f) => return f,
        };
    let collection_id = p.id.clone();
    let collection = PromptCollection {
        id: p.id,
        name: p.name,
        description: p.description,
        documents: p.documents,
    };
    match git_blocking(request_id, state, move |git| {
        crate::agent::context::upsert_prompt_collection(&git, &collection)
    })
    .await
    {
        Ok(()) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptCollectionCreateResponse { id: collection_id },
        ),
        Err(frame) => frame,
    }
}

/// Handle `prompt.collection.list` requests.
pub(super) async fn handle_collection_list(request_id: &str, state: &SharedState) -> Frame {
    match git_blocking(request_id, state, |git| {
        crate::agent::context::list_prompt_collections(&git)
    })
    .await
    {
        Ok(cols) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptCollectionListResponse { collections: cols },
        ),
        Err(frame) => frame,
    }
}

/// Handle `prompt.collection.delete` requests.
pub(super) async fn handle_collection_delete(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PromptCollectionDeleteParams =
        match parse_params(request_id, params, "prompt.collection.delete") {
            Ok(v) => v,
            Err(f) => return f,
        };
    match git_blocking(request_id, state, move |git| {
        crate::agent::context::delete_prompt_collection(&git, &p.id)
    })
    .await
    {
        Ok(deleted) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PromptCollectionDeleteResponse { deleted },
        ),
        Err(frame) => frame,
    }
}
