use crate::error::{Error, Result};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use nexo_core::ClientKind;
use nexo_ws_schema::{
    Frame, GatewayToNodeMessage, GatewayToUserMessage, NodeToGatewayMessage, UserToGatewayMessage,
};
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tracing::{debug, error, info, trace, warn};
type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// A WebSocket connection to a NEXO Gateway.
pub struct NexoConnection {
    ws: WsStream,
}

/// Write half of a split [`NexoConnection`].
pub struct WriteHalf {
    sink: SplitSink<WsStream, Message>,
}

/// Read half of a split [`NexoConnection`].
pub struct ReadHalf {
    stream: SplitStream<WsStream>,
}

impl NexoConnection {
    /// Connect to a gateway at the given URL with the auth header.
    pub async fn connect(url: &str, auth_token: &str, client_kind: ClientKind) -> Result<Self> {
        // Build a WebSocket request with the given URL and auth token.
        let request = http::Request::builder()
            .uri(url)
            .header(nexo_ws_schema::AUTH_HEADER, auth_token)
            .header("Host", host_from_url(url))
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
            .map_err(|e| Error::Handshake(format!("Failed to build request: {e}")))?;

        let (mut ws, _response) = tokio_tungstenite::connect_async(request).await?;
        debug!("WebSocket connection established to {url}");

        // Perform the handshake with the gateway, sending the connect message
        match client_kind {
            ClientKind::User(properties) => {
                debug!("Performing User handshake");
                let register_frame = Frame::new(UserToGatewayMessage::Connect(properties))?;
                send_frame_impl(&mut ws, &register_frame).await?;

                loop {
                    match recv_frame_impl(&mut ws).await {
                        Ok(frame) => {
                            let (frame_id, payload) = frame.into_parts::<GatewayToUserMessage>()?;
                            debug!(?frame_id, "Received frame");

                            match payload {
                                GatewayToUserMessage::Connect(response) => {
                                    response.is_completed()?;
                                    break;
                                }
                                _ => {
                                    warn!(payload = ?payload, "Unexpected handshake response, ignoring...");
                                }
                            };
                        }
                        Err(e) => {
                            error!(error = ?e, "Failed to receive handshake response");
                            return Err(e);
                        }
                    };
                }
            }
            ClientKind::Node(properties) => {
                debug!("Performing Node handshake");
                let register_frame = Frame::new(NodeToGatewayMessage::Connect(properties))?;
                send_frame_impl(&mut ws, &register_frame).await?;

                loop {
                    match recv_frame_impl(&mut ws).await {
                        Ok(frame) => {
                            let (frame_id, payload) = frame.into_parts::<GatewayToNodeMessage>()?;
                            debug!(?frame_id, "Received frame");

                            match payload {
                                GatewayToNodeMessage::Connect(response) => {
                                    response.is_completed()?;
                                    break;
                                }
                                _ => {
                                    warn!(payload = ?payload, "Unexpected handshake response, ignoring...");
                                }
                            };
                        }
                        Err(e) => {
                            error!(error = ?e, "Failed to receive handshake response");
                            return Err(e);
                        }
                    };
                }
            }
        }

        info!("Handshake successful");
        Ok(Self { ws })
    }

    /// Send a typed Frame as a JSON text message.
    #[inline]
    pub async fn send_frame(&mut self, frame: &Frame) -> Result {
        send_frame_impl(&mut self.ws, frame).await
    }

    /// Receive the next Frame. Returns an error if the connection is closed.
    #[inline]
    pub async fn recv_frame(&mut self) -> Result<Frame> {
        recv_frame_impl(&mut self.ws).await
    }

    /// Close the connection gracefully.
    pub async fn close(&mut self) -> Result {
        close_impl(&mut self.ws).await
    }

    /// Split this connection into independent read and write halves.
    pub fn into_split(self) -> (WriteHalf, ReadHalf) {
        let (sink, stream) = self.ws.split();
        (WriteHalf { sink }, ReadHalf { stream })
    }
}

impl WriteHalf {
    /// Send a typed Frame as a JSON text message.
    #[inline]
    pub async fn send_frame(&mut self, frame: &Frame) -> Result {
        send_frame_impl(&mut self.sink, frame).await
    }

    /// Close the write half gracefully.
    #[inline]
    pub async fn close(&mut self) -> Result {
        close_impl(&mut self.sink).await
    }
}

impl ReadHalf {
    /// Receive the next Frame. Returns an error if the connection is closed.
    #[inline]
    pub async fn recv_frame(&mut self) -> Result<Frame> {
        recv_frame_impl(&mut self.stream).await
    }
}

/// Generic implementation for sending a message from connection or write half.
#[inline]
async fn send_frame_impl<S>(sink: &mut S, frame: &Frame) -> Result
where
    S: Sink<Message, Error = WsError> + Unpin,
{
    let json = serde_json::to_string(frame)?;
    trace!(">>> {json}");
    sink.send(Message::Text(json.into())).await?;
    Ok(())
}

/// Generic implementation for receiving a message from connection or read half.
#[inline]
async fn recv_frame_impl<St>(stream: &mut St) -> Result<Frame>
where
    St: Stream<Item = std::result::Result<Message, WsError>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                trace!("<<< {text}");
                let frame: Frame = serde_json::from_str(&text)?;
                return Ok(frame);
            }
            Some(Ok(Message::Close(_))) | None => {
                return Err(Error::Closed);
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {
                // Ignore control frames, continue reading
                continue;
            }
            Some(Ok(Message::Binary(_))) => {
                warn!("Received unexpected binary message, ignoring");
                continue;
            }
            Some(Err(e)) => return Err(e.into()),
        }
    }
}

/// Generic implementation for closing a connection or write half.
async fn close_impl<S>(sink: &mut S) -> Result
where
    S: Sink<Message, Error = WsError> + Unpin,
{
    sink.close().await?;
    Ok(())
}

/// Extract the host from a WebSocket URL for the Host header.
fn host_from_url(url: &str) -> String {
    url.strip_prefix("ws://")
        .or_else(|| url.strip_prefix("wss://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("localhost")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_from_url_parses_correctly() {
        assert_eq!(host_from_url("ws://127.0.0.1:6969"), "127.0.0.1:6969");
        assert_eq!(host_from_url("wss://example.com/path"), "example.com");
        assert_eq!(host_from_url("localhost:8080"), "localhost:8080");
    }
}
