use nexo_core::{NexoClient, UserProperties};
use nexo_ws_client::{NexoConnection, ReadHalf, WriteHalf};
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
    user: &UserProperties,
) -> cli_helpers::Result<(
    ConnectionInfo,
    UnboundedSender<NetworkCommand>,
    UnboundedReceiver<NetworkEvent>,
)> {
    let url = user.gateway_url().to_string();
    let conn = NexoConnection::connect(&url, NexoClient::User(user.clone()))
        .await
        .map_err(|e| cli_helpers::Error::Other(format!("Connection failed: {e}")))?;

    let (writer, reader) = conn.into_split();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    tokio::spawn(writer_loop(writer, command_rx, event_tx.clone()));
    tokio::spawn(reader_loop(reader, event_tx));

    Ok((
        ConnectionInfo {
            gateway_url: url,
            protocol: user.protocol().max_protocol,
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
            Ok(frame) => {
                let _ = event_tx.send(NetworkEvent::Frame(frame));
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
