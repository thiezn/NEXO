use cli_helpers::Error;
use nexo_ai::coordinator::Coordinator;
use nexo_ws_client::WriteHalf;
use nexo_ws_schema::{ErrorPayload, Frame, LoadedModelInfo, Method, ModelStatusParams};
use std::sync::{Arc, Mutex, MutexGuard};

fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Result<MutexGuard<'a, T>, String> {
    mutex
        .lock()
        .map_err(|error| format!("Failed to lock {name}: {error}"))
}

/// Send a frame, mapping websocket errors into `Error::Network`.
pub(crate) async fn send(writer: &mut WriteHalf, frame: &Frame) -> cli_helpers::Result {
    writer
        .send_frame(frame)
        .await
        .map_err(|error| Error::Network(format!("Send error: {error}")))
}

/// Send a `node_busy` error response while another inference is in progress.
pub(crate) async fn send_busy_error(
    writer: &mut WriteHalf,
    request_id: &str,
) -> cli_helpers::Result {
    let error = Frame::error_response(
        request_id,
        ErrorPayload {
            code: "node_busy".into(),
            message: "Inference is already in progress".into(),
        },
    );
    send(writer, &error).await
}

/// Push the node's current loaded model state to the gateway.
pub(crate) async fn push_model_status(
    writer: &mut WriteHalf,
    coordinator: &Arc<Mutex<Coordinator>>,
    available_models: &[String],
) {
    let loaded_models: Vec<LoadedModelInfo> = {
        let coord = match lock_mutex(coordinator.as_ref(), "coordinator") {
            Ok(coord) => coord,
            Err(error) => {
                tracing::warn!("Failed to build model status update: {error}");
                return;
            }
        };
        coord
            .loaded_models()
            .iter()
            .map(|(name, categories)| LoadedModelInfo {
                model_id: name.to_string(),
                categories: categories.to_vec(),
            })
            .collect()
    };

    let status = ModelStatusParams {
        loaded_models,
        available_models: available_models.to_vec(),
    };
    if let Ok(frame) = Frame::request(Method::ModelStatus, &status) {
        let _ = writer.send_frame(&frame).await;
    }
}
