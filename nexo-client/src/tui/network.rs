use crate::config::ClientConfig;
use nexo_ws_client::{NexoConnection, ReadHalf, WriteHalf, perform_handshake};
use nexo_ws_schema::Frame;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use super::model::ConnectionInfo;

#[derive(Debug)]
pub enum NetworkCommand {
    Send(Frame),
    Close,
}

#[derive(Debug)]
pub enum NetworkEvent {
    Frame(Frame),
    Disconnected(String),
}

pub async fn connect(
    url_override: Option<&str>,
) -> cli_helpers::Result<(
    ConnectionInfo,
    UnboundedSender<NetworkCommand>,
    UnboundedReceiver<NetworkEvent>,
)> {
    let config = ClientConfig::load()?;
    let url = url_override
        .map(str::to_owned)
        .unwrap_or_else(|| config.gateway_url.clone());

    let mut conn = NexoConnection::connect(&url, &config.auth_token)
        .await
        .map_err(|e| cli_helpers::Error::Network(format!("Connection failed: {e}")))?;

    let params = nexo_ws_client::default_user_connect_params(
        &config.client_id,
        &config.client_version,
        config.platform,
        &config.device_id,
    );
    let hello = perform_handshake(&mut conn, params)
        .await
        .map_err(|e| cli_helpers::Error::Network(format!("Handshake failed: {e}")))?;

    let (writer, reader) = conn.into_split();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    tokio::spawn(writer_loop(writer, command_rx, event_tx.clone()));
    tokio::spawn(reader_loop(reader, event_tx));

    Ok((
        ConnectionInfo {
            gateway_url: url,
            hello,
        },
        command_tx,
        event_rx,
    ))
}

async fn writer_loop(
    mut writer: WriteHalf,
    mut command_rx: UnboundedReceiver<NetworkCommand>,
    event_tx: UnboundedSender<NetworkEvent>,
) {
    while let Some(command) = command_rx.recv().await {
        match command {
            NetworkCommand::Send(frame) => {
                if let Err(error) = writer.send_frame(&frame).await {
                    let _ =
                        event_tx.send(NetworkEvent::Disconnected(format!("Send failed: {error}")));
                    break;
                }
            }
            NetworkCommand::Close => {
                let _ = writer.close().await;
                break;
            }
        }
    }
}

async fn reader_loop(mut reader: ReadHalf, event_tx: UnboundedSender<NetworkEvent>) {
    loop {
        match reader.recv_frame().await {
            Ok(Some(frame)) => {
                let _ = event_tx.send(NetworkEvent::Frame(frame));
            }
            Ok(None) => {
                let _ = event_tx.send(NetworkEvent::Disconnected(
                    "Connection closed by server".to_string(),
                ));
                break;
            }
            Err(error) => {
                let _ = event_tx.send(NetworkEvent::Disconnected(format!(
                    "Receive failed: {error}"
                )));
                break;
            }
        }
    }
}
