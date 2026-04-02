use crate::config::ClientConfig;
use nexo_ws_client::{NexoConnection, perform_handshake};
use nexo_ws_schema::{Frame, HelloOk};

/// Load config, connect to gateway, and perform handshake.
pub async fn connect_and_handshake(
    url_override: Option<&str>,
) -> utl_helpers::Result<(NexoConnection, HelloOk)> {
    let config = ClientConfig::load()?;
    let url = url_override
        .map(String::from)
        .unwrap_or_else(|| config.gateway_url.clone());

    tracing::info!("Connecting to {url}...");

    let mut conn = NexoConnection::connect(&url, &config.auth_token)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Connection failed: {e}")))?;

    let params = nexo_ws_client::default_user_connect_params(
        &config.client_id,
        &config.client_version,
        config.platform,
        &config.device_id,
    );

    let hello = perform_handshake(&mut conn, params)
        .await
        .map_err(|e| utl_helpers::Error::Network(format!("Handshake failed: {e}")))?;

    tracing::info!(
        "Connected! Protocol v{}, tick interval {}ms",
        hello.protocol,
        hello.policy.tick_interval_ms
    );

    Ok((conn, hello))
}

/// Receive frames until we get a response (skipping events).
pub async fn recv_response(conn: &mut NexoConnection) -> utl_helpers::Result<serde_json::Value> {
    loop {
        let frame = conn
            .recv_frame()
            .await
            .map_err(|e| utl_helpers::Error::Network(format!("Receive error: {e}")))?
            .ok_or_else(|| utl_helpers::Error::Network("Connection closed".to_string()))?;

        match frame {
            Frame::Response {
                ok, payload, error, ..
            } => {
                if !ok {
                    let msg = error
                        .map(|e| format!("{}: {}", e.code, e.message))
                        .unwrap_or_else(|| "Unknown error".to_string());
                    return Err(utl_helpers::Error::Network(msg));
                }
                return Ok(payload.unwrap_or(serde_json::Value::Null));
            }
            Frame::Event { .. } => continue,
            Frame::Request { .. } => {
                tracing::debug!("Received server-initiated request (ignoring)");
                continue;
            }
        }
    }
}
