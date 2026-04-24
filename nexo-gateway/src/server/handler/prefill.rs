use crate::server::state::SharedState;
use nexo_ws_schema::{
    ErrorPayload, Frame, PrefillCollectionCreateParams, PrefillCollectionDeleteParams,
    PrefillMarkdownCreateParams, PrefillMarkdownDeleteParams,
};

use super::base::{git_blocking, ok_or_internal_error, parse_params};

pub(super) fn handle_fetch_deprecated(request_id: &str) -> Frame {
    Frame::error_response(
        request_id,
        ErrorPayload {
            code: "deprecated".into(),
            message:
                "prefill.fetch is no longer used; prefill content is included in the system prompt"
                    .into(),
        },
    )
}

pub(super) async fn handle_markdown_create(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PrefillMarkdownCreateParams =
        match parse_params(request_id, params, "prefill.markdown.create") {
            Ok(v) => v,
            Err(f) => return f,
        };
    let filename = p.filename.clone();
    match git_blocking(request_id, state, move |git| {
        crate::agent::prefill::create_markdown(&git, &p.filename, &p.content)
    })
    .await
    {
        Ok(()) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillMarkdownCreateResponse { filename },
        ),
        Err(frame) => frame,
    }
}

pub(super) async fn handle_markdown_list(request_id: &str, state: &SharedState) -> Frame {
    match git_blocking(request_id, state, |git| {
        crate::agent::prefill::list_markdown(&git)
    })
    .await
    {
        Ok(files) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillMarkdownListResponse {
                files: files
                    .into_iter()
                    .map(|f| nexo_ws_schema::MarkdownFileEntry {
                        filename: f.filename,
                    })
                    .collect(),
            },
        ),
        Err(frame) => frame,
    }
}

pub(super) async fn handle_markdown_delete(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PrefillMarkdownDeleteParams =
        match parse_params(request_id, params, "prefill.markdown.delete") {
            Ok(v) => v,
            Err(f) => return f,
        };
    match git_blocking(request_id, state, move |git| {
        crate::agent::prefill::delete_markdown(&git, &p.filename)
    })
    .await
    {
        Ok(deleted) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillMarkdownDeleteResponse { deleted },
        ),
        Err(frame) => frame,
    }
}

pub(super) async fn handle_collection_create(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PrefillCollectionCreateParams =
        match parse_params(request_id, params, "prefill.collection.create") {
            Ok(v) => v,
            Err(f) => return f,
        };
    let collection_id = p.id.clone();
    match git_blocking(request_id, state, move |git| {
        crate::agent::prefill::create_collection(
            &git,
            &p.id,
            &p.name,
            p.description.as_deref(),
            &p.markdown_files,
        )
    })
    .await
    {
        Ok(()) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillCollectionCreateResponse { id: collection_id },
        ),
        Err(frame) => frame,
    }
}

pub(super) async fn handle_collection_list(request_id: &str, state: &SharedState) -> Frame {
    match git_blocking(request_id, state, |git| {
        crate::agent::prefill::list_collections(&git)
    })
    .await
    {
        Ok(cols) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillCollectionListResponse {
                collections: cols
                    .into_iter()
                    .map(|c| nexo_ws_schema::CollectionEntry {
                        id: c.id,
                        name: c.name,
                        description: c.description,
                        markdown_files: c.markdown_files,
                    })
                    .collect(),
            },
        ),
        Err(frame) => frame,
    }
}

pub(super) async fn handle_collection_delete(
    request_id: &str,
    params: serde_json::Value,
    state: &SharedState,
) -> Frame {
    let p: PrefillCollectionDeleteParams =
        match parse_params(request_id, params, "prefill.collection.delete") {
            Ok(v) => v,
            Err(f) => return f,
        };
    match git_blocking(request_id, state, move |git| {
        crate::agent::prefill::delete_collection(&git, &p.id)
    })
    .await
    {
        Ok(deleted) => ok_or_internal_error(
            request_id,
            nexo_ws_schema::PrefillCollectionDeleteResponse { deleted },
        ),
        Err(frame) => frame,
    }
}
