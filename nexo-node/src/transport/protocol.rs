use cli_helpers::Error;
use nexo_ws_client::WriteHalf;
use nexo_ws_schema::{ErrorPayload, Frame};

/// Send a frame, mapping websocket errors into the CLI error type.
pub(crate) async fn send(writer: &mut WriteHalf, frame: &Frame) -> cli_helpers::Result {
    writer
        .send_frame(frame)
        .await
        .map_err(|error| Error::Other(format!("Send error: {error}")))
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
