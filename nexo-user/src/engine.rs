use crate::Result;
use nexo_core::{NexoClient, UserProperties};
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::{Frame, GatewayToUserMessage, UserToGatewayMessage};
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tracing::{debug, error, info, warn};
/// Central coordinator for nexo-user, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub struct NexoUser {
    /// The configuration for the user, including gateway URL, auth token, and node identity.
    config: UserProperties,
}

impl NexoUser {
    /// Initializes a new NexoUser from prepared properties and runtime dependencies.
    pub fn new(config: UserProperties) -> Self {
        Self { config }
    }

    /// Connect to the nexo-gateway.
    async fn connect(&self) -> Result<NexoConnection> {
        let url = self.config.gateway_url();
        info!(url = url, "Connecting to gateway...");

        let conn = NexoConnection::connect(url, NexoClient::User(self.config.clone())).await?;

        info!("User setup complete, entering main loop");
        Ok(conn)
    }

    /// Start the NexoUser runtime, connect to the gateway, and begin processing messages.
    pub async fn run(&self) -> Result {
        let mut conn = self.connect().await?;

        let (tx, mut rx) = mpsc::channel::<UserToGatewayMessage>(100);

        loop {
            tokio::select! {

                    // Handle incoming frames from the gateway
                    frame = conn.recv_frame() => {
                        match frame {
                            Ok(frame) => {
                                self.handle_frame(frame, tx.clone()).await?;
                            }
                            Err(e) => {
                                error!("Websocket receive error: {e}");
                                return Err(e.into());
                            }
                        }
                    }

                    // Handle results of actions taken in response to gateway messages
                    Some(msg) = rx.recv() => {

                        if matches!(msg, UserToGatewayMessage::Disconnect(_)) {
                            info!("Disconnecting from gateway...");
                            let frame = Frame::new(msg)?;
                            conn.send_frame(&frame).await?;
                            break;
                        }

                        let frame = Frame::new(msg)?;
                        conn.send_frame(&frame).await?;
                    }
            }
        }

        if let Err(error) = conn.close().await {
            debug!("Close error (non-fatal): {error}");
        }

        Ok(())
    }

    /// Handle an incoming Frame from the gateway.
    ///
    /// The loop does not process the next frame until this function returns,
    /// so any long-running operations should be offloaded to a separate task.
    ///
    /// The obvious candidates for offloading are inference and tool call operations, which
    /// can take a long time to complete.
    async fn handle_frame(&self, frame: Frame, tx: Sender<UserToGatewayMessage>) -> Result {
        let (frame_id, payload) = frame.into_parts::<GatewayToUserMessage>()?;
        info!(frame_id = ?frame_id, "Received frame");

        // Guidelines for handling incoming messages:
        //
        // - Messages that are informational only, we can leverage the response.result() helper to log the outcome and return early.
        // - Messages that do not need a reply and can be handled 'immediately' can be handled inline.
        // - Messages that require a response and are long-running should be offloaded to a separate task. Make sure to first send
        //   an Accepted response back to the gateway before offloading the work to a task. This will ensure the gateway knows the request is
        //   being processed and knows to expect follow up events.
        match payload {
            GatewayToUserMessage::Disconnect(response) => {
                let _ = response.result();
            }
            GatewayToUserMessage::GetState(_) => {
                todo!("GetState request handling not implemented yet");
            }
            GatewayToUserMessage::ClearSession(_) => {
                todo!("ClearSession request handling not implemented yet");
            }
            GatewayToUserMessage::ListSessions(_) => {
                todo!("ListSessions request handling not implemented yet");
            }
            GatewayToUserMessage::GetSession(_) => {
                todo!("GetSession request handling not implemented yet");
            }
            GatewayToUserMessage::StartInferenceRun(_) => {
                todo!("StartInferenceRun request handling not implemented yet");
            }
            GatewayToUserMessage::StartInferenceRunEvent(_) => {
                todo!("StartInferenceRunEvent request handling not implemented yet");
            }
            GatewayToUserMessage::Cancel(_) => {
                todo!(
                    "Cancel request handling not implemented yet. I think I want to remove the generic cancel request in favor of specific ones. This will ensure the request can contain all required information to make the call instead of maintaining that state in the nexo node memory."
                );
            }
            GatewayToUserMessage::Connect(_) => {
                let name: &'static str = (&payload).into();
                warn!(
                    name = name,
                    "Received unexpected message from gateway, ignoring"
                );
            }
        };

        Ok(())
    }
}
