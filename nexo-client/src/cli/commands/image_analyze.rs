use super::helpers::{connect_and_handshake, recv_response};
use base64::Engine;
use nexo_ws_schema::{Frame, ImageAnalyzeParams, ImageAnalyzeResponse, Method};

pub async fn run_image_analyze(image_path: String, prompt: String) -> utl_helpers::Result {
    let image_bytes = std::fs::read(&image_path)
        .map_err(|e| utl_helpers::Error::Io(format!("Failed to read image '{image_path}': {e}")))?;

    tracing::info!(
        "Read {} bytes from '{image_path}'",
        image_bytes.len()
    );

    let image_data = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    let (mut conn, _hello) = connect_and_handshake(None).await?;

    let analyze_params = ImageAnalyzeParams {
        image_data,
        prompt,
        max_tokens: 4096,
        temperature: 1.0,
        visual_token_budget: None,
        idempotency_key: Frame::new_id(),
    };

    let frame = Frame::request(Method::ImageAnalyze, &analyze_params)
        .map_err(|e| utl_helpers::Error::Network(format!("Failed to build request: {e}")))?;

    conn.send_frame(&frame)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Send failed: {e}")))?;

    tracing::info!("Waiting for analysis response...");

    let response = recv_response(&mut conn).await?;
    let result: ImageAnalyzeResponse = serde_json::from_value(response)
        .map_err(|e| utl_helpers::Error::Network(format!("Invalid response: {e}")))?;

    println!("{}", result.text);
    println!(
        "\n[{} tokens in {}ms]",
        result.tokens_generated, result.inference_time_ms
    );

    if let Err(e) = conn.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }

    Ok(())
}
