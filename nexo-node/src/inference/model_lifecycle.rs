use crate::inference::SessionCacheManager;
use crate::transport::{push_model_status, send};
use cli_helpers::Error;
use nexo_ai::coordinator::Coordinator;
use nexo_ai::registry::find_manifest;
use nexo_ws_client::WriteHalf;
use nexo_ws_schema::{ErrorPayload, Frame, ModelLoadResponse, ModelUnloadResponse};
use serde::Deserialize;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Deserialize)]
struct ModelIdParams {
    #[serde(rename = "modelId")]
    model_id: String,
}

fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Result<MutexGuard<'a, T>, String> {
    mutex
        .lock()
        .map_err(|error| format!("Failed to lock {name}: {error}"))
}

/// Load a model in response to a gateway request and refresh advertised status.
pub(crate) async fn handle_model_load(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
) -> cli_helpers::Result {
    let model_id = match parse_required_model_id(params) {
        Ok(model_id) => model_id,
        Err(message) => {
            send_invalid_params_error(writer, request_id, message).await?;
            return Ok(());
        }
    };

    tracing::info!("Loading model '{model_id}'");

    let coord = coordinator.clone();
    let model_id_clone = model_id.clone();
    let (loaded, error) =
        tokio::task::spawn_blocking(move || match lock_mutex(coord.as_ref(), "coordinator") {
            Ok(mut coord) => match coord.load_model(&model_id_clone) {
                Ok(()) => (true, None),
                Err(error) => {
                    tracing::error!("Failed to load model '{model_id_clone}': {error}");
                    (false, Some(error.to_string()))
                }
            },
            Err(error) => {
                tracing::error!("Failed to load model '{model_id_clone}': {error}");
                (false, Some(error))
            }
        })
        .await
        .unwrap_or((false, Some("Task panicked".into())));

    let response = Frame::ok_response(
        request_id,
        &ModelLoadResponse {
            model_id: model_id.clone(),
            loaded,
            error,
        },
    )
    .unwrap_or_else(|error| {
        Frame::error_response(
            request_id,
            ErrorPayload {
                code: "internal_error".into(),
                message: error.to_string(),
            },
        )
    });

    send(writer, &response).await?;

    if loaded {
        if let Some(manifest) = find_manifest(&model_id) {
            let mut coord =
                lock_mutex(coordinator.as_ref(), "coordinator").map_err(Error::Other)?;
            for category in &manifest.categories {
                coord.set_active_model(*category, model_id.clone());
            }
        }
        push_model_status(writer, coordinator, available_models).await;
    }

    Ok(())
}

/// Unload a model in response to a gateway request and persist its KV cache when possible.
pub(crate) async fn handle_model_unload(
    writer: &mut WriteHalf,
    request_id: &str,
    params: serde_json::Value,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
    cache_manager: &Arc<Mutex<SessionCacheManager>>,
) -> cli_helpers::Result {
    let model_id = match parse_required_model_id(params) {
        Ok(model_id) => model_id,
        Err(message) => {
            send_invalid_params_error(writer, request_id, message).await?;
            return Ok(());
        }
    };

    tracing::info!("Unloading model '{model_id}'");

    let coord = coordinator.clone();
    let cache_mgr = cache_manager.clone();
    let model_id_clone = model_id.clone();
    let unloaded = tokio::task::spawn_blocking(move || {
        let mut coord = match lock_mutex(coord.as_ref(), "coordinator") {
            Ok(coord) => coord,
            Err(error) => {
                tracing::error!("Failed to unload model '{model_id_clone}': {error}");
                return false;
            }
        };

        if let Some(model) = coord.model_mut(&model_id_clone)
            && let Some(kv) = model.as_kv_cacheable()
        {
            match lock_mutex(cache_mgr.as_ref(), "session cache manager") {
                Ok(manager) => {
                    if let Err(error) = manager.on_model_unload(&model_id_clone, kv) {
                        tracing::warn!("Failed to save KV cache before unload: {error}");
                    }
                }
                Err(error) => {
                    tracing::warn!("Failed to save KV cache before unload: {error}");
                }
            }
        }

        match coord.unload_model(&model_id_clone) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!("Unload of model '{model_id_clone}' failed (non-fatal): {error}");
                false
            }
        }
    })
    .await
    .unwrap_or(false);

    let response = Frame::ok_response(request_id, &ModelUnloadResponse { unloaded })
        .unwrap_or_else(|error| {
            Frame::error_response(
                request_id,
                ErrorPayload {
                    code: "internal_error".into(),
                    message: error.to_string(),
                },
            )
        });

    send(writer, &response).await?;

    if unloaded && let Some(manifest) = find_manifest(&model_id) {
        let mut coord = lock_mutex(coordinator.as_ref(), "coordinator").map_err(Error::Other)?;
        for category in &manifest.categories {
            if coord
                .active_model_for(*category)
                .is_some_and(|active| active == model_id)
            {
                coord.remove_active_model(*category);
            }
        }
    }

    push_model_status(writer, coordinator, available_models).await;

    Ok(())
}

fn parse_required_model_id(params: serde_json::Value) -> Result<String, String> {
    let params: ModelIdParams = serde_json::from_value(params)
        .map_err(|_| "Expected params with a string 'modelId' field".to_string())?;
    let model_id = params.model_id.trim();

    if model_id.is_empty() {
        return Err("Parameter 'modelId' must not be empty".to_string());
    }

    Ok(model_id.to_string())
}

async fn send_invalid_params_error(
    writer: &mut WriteHalf,
    request_id: &str,
    message: String,
) -> cli_helpers::Result {
    let error_response = Frame::error_response(
        request_id,
        ErrorPayload {
            code: "invalid_params".into(),
            message,
        },
    );

    send(writer, &error_response).await
}

#[cfg(test)]
mod tests {
    use super::parse_required_model_id;
    use serde_json::json;

    #[test]
    fn parses_model_id_from_typed_params() {
        let model_id = parse_required_model_id(json!({ "modelId": "llama-3.2" }));

        assert_eq!(model_id, Ok("llama-3.2".to_string()));
    }

    #[test]
    fn rejects_missing_model_id() {
        let error = parse_required_model_id(json!({}));

        assert!(matches!(&error, Err(message) if message.contains("modelId")));
    }

    #[test]
    fn rejects_empty_model_id() {
        let error = parse_required_model_id(json!({ "modelId": "   " }));

        assert_eq!(
            error,
            Err("Parameter 'modelId' must not be empty".to_string())
        );
    }
}
