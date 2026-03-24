use crate::config::ClientConfig;
use nexo_ws_client::{NexoConnection, perform_handshake};
use nexo_ws_schema::{Frame, Method};
use tokio::io::{AsyncBufReadExt, BufReader};

pub async fn run_connect(url_override: Option<String>) -> utl_helpers::Result {
    let config = ClientConfig::load()?;
    let url = url_override.unwrap_or_else(|| config.gateway_url.clone());

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

    println!("Connected to {url}");
    println!("Type a method (health, status, tools.catalog) or 'quit' to exit:");

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        tokio::select! {
            line = lines.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        let line = line.trim().to_string();
                        if line == "quit" || line == "exit" {
                            break;
                        }
                        if line.is_empty() {
                            continue;
                        }
                        if let Err(e) = handle_user_input(&mut conn, &line).await {
                            tracing::error!("Send error: {e}");
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("Stdin error: {e}");
                        break;
                    }
                }
            }
            frame = conn.recv_frame() => {
                match frame {
                    Ok(Some(frame)) => {
                        match serde_json::to_string_pretty(&frame) {
                            Ok(json) => println!("<<< {json}"),
                            Err(e) => tracing::error!("Failed to format frame: {e}"),
                        }
                    }
                    Ok(None) => {
                        tracing::info!("Connection closed by server");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("Receive error: {e}");
                        break;
                    }
                }
            }
        }
    }

    if let Err(e) = conn.close().await {
        tracing::debug!("Close error (non-fatal): {e}");
    }
    println!("Disconnected.");
    Ok(())
}

async fn handle_user_input(
    conn: &mut NexoConnection,
    input: &str,
) -> std::result::Result<(), String> {
    let method = match input {
        "health" => Method::Health,
        "status" => Method::Status,
        "tools.catalog" => Method::ToolsCatalog,
        "system-presence" => Method::SystemPresence,
        _ => {
            println!("Unknown method: {input}. Available: health, status, tools.catalog, system-presence");
            return Ok(());
        }
    };

    let params = match method {
        Method::SystemPresence => serde_json::json!({"status": "active"}),
        _ => serde_json::json!({}),
    };

    let frame = Frame::request(method, params).map_err(|e| format!("Failed to build request: {e}"))?;

    match serde_json::to_string_pretty(&frame) {
        Ok(json) => println!(">>> {json}"),
        Err(e) => tracing::error!("Failed to format frame: {e}"),
    }

    conn.send_frame(&frame)
        .await
        .map_err(|e| format!("Send failed: {e}"))
}
