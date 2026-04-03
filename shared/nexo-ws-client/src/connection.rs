use crate::error::{ClientError, Result};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use nexo_ws_schema::Frame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
    pub async fn connect(url: &str, auth_token: &str) -> Result<Self> {
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
            .map_err(|e| ClientError::Handshake(format!("Failed to build request: {e}")))?;

        let (ws, _response) = tokio_tungstenite::connect_async(request).await?;
        tracing::debug!("WebSocket connection established to {url}");
        Ok(Self { ws })
    }

    /// Send a typed Frame as a JSON text message.
    pub async fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let json = serde_json::to_string(frame)?;
        tracing::trace!(">>> {json}");
        self.ws.send(Message::Text(json.into())).await?;
        Ok(())
    }

    /// Receive the next Frame. Returns `None` if the connection is closed.
    pub async fn recv_frame(&mut self) -> Result<Option<Frame>> {
        loop {
            match self.ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    tracing::trace!("<<< {text}");
                    let frame: Frame = serde_json::from_str(&text)?;
                    return Ok(Some(frame));
                }
                Some(Ok(Message::Close(_))) | None => {
                    return Ok(None);
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {
                    // Ignore control frames, continue reading
                    continue;
                }
                Some(Ok(Message::Binary(_))) => {
                    tracing::warn!("Received unexpected binary message, ignoring");
                    continue;
                }
                Some(Err(e)) => return Err(e.into()),
            }
        }
    }

    /// Send a raw JSON string (for testing/debugging).
    pub(crate) async fn send_raw(&mut self, json: &str) -> Result<()> {
        self.ws
            .send(Message::Text(json.to_string().into()))
            .await?;
        Ok(())
    }

    /// Close the connection gracefully.
    pub async fn close(&mut self) -> Result<()> {
        self.ws.close(None).await?;
        Ok(())
    }

    /// Split this connection into independent read and write halves.
    pub fn into_split(self) -> (WriteHalf, ReadHalf) {
        let (sink, stream) = self.ws.split();
        (WriteHalf { sink }, ReadHalf { stream })
    }
}

impl WriteHalf {
    /// Send a typed Frame as a JSON text message.
    pub async fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let json = serde_json::to_string(frame)?;
        tracing::trace!(">>> {json}");
        self.sink.send(Message::Text(json.into())).await?;
        Ok(())
    }

    /// Close the write half gracefully.
    pub async fn close(&mut self) -> Result<()> {
        self.sink.close().await?;
        Ok(())
    }
}

impl ReadHalf {
    /// Receive the next Frame. Returns `None` if the connection is closed.
    pub async fn recv_frame(&mut self) -> Result<Option<Frame>> {
        loop {
            match self.stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    tracing::trace!("<<< {text}");
                    let frame: Frame = serde_json::from_str(&text)?;
                    return Ok(Some(frame));
                }
                Some(Ok(Message::Close(_))) | None => {
                    return Ok(None);
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {
                    continue;
                }
                Some(Ok(Message::Binary(_))) => {
                    tracing::warn!("Received unexpected binary message, ignoring");
                    continue;
                }
                Some(Err(e)) => return Err(e.into()),
            }
        }
    }
}

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
        assert_eq!(
            host_from_url("wss://example.com/path"),
            "example.com"
        );
        assert_eq!(host_from_url("localhost:8080"), "localhost:8080");
    }
}
