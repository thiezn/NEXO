use crate::{Result, TuiAction, TuiController, TuiEvent};
use nexo_core::{NexoClient, UserProperties};
use nexo_ws_client::NexoConnection;
use nexo_ws_schema::{Frame, GatewayToUserMessage};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::{debug, error, info};

/// Central coordinator for nexo-user, ties configuration,
/// tool registry, websocket loop and inference engine together.
pub struct NexoUser {
    /// The configuration for the user, including gateway URL, auth token, and node identity.
    config: UserProperties,
}

impl NexoUser {
    /// Initializes a new NexoUser from prepared properties and runtime dependencies.
    ///
    /// # Arguments
    ///
    /// * `config` - The user properties used to connect to and identify with the gateway.
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
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the websocket loop and TUI loop shut down cleanly.
    pub async fn run(&self) -> Result {
        let mut conn = self.connect().await?;

        let (action_tx, action_rx) = mpsc::channel::<TuiAction>(100);
        let (event_tx, mut event_rx) = mpsc::channel::<TuiEvent>(100);
        let mut tui_controller = TuiController::new();

        event_tx.send(TuiEvent::Connected).await?;

        let tui_task =
            tokio::spawn(async move { tui_controller.run(action_tx, &mut event_rx).await });

        self.send_action(&mut conn, TuiAction::RefreshState).await?;

        self.run_event_loop(conn, action_rx, event_tx).await?;

        tui_task
            .await
            .map_err(|error| crate::Error::Other(format!("TUI task failed to join: {error}")))??;

        Ok(())
    }

    /// Runs the coordinated websocket and action processing loop.
    ///
    /// # Arguments
    ///
    /// * `conn` - The active websocket connection to the gateway.
    /// * `action_rx` - The receiver used to consume UI actions.
    /// * `event_tx` - The sender used to publish TUI events.
    async fn run_event_loop(
        &self,
        mut conn: NexoConnection,
        mut action_rx: Receiver<TuiAction>,
        event_tx: Sender<TuiEvent>,
    ) -> Result {
        loop {
            tokio::select! {

                    frame = conn.recv_frame() => {
                        match frame {
                            Ok(frame) => {
                                self.handle_frame(frame, &event_tx).await?;
                            }
                            Err(e) => {
                                error!(error = %e, "Websocket receive error");
                                let _ = event_tx.send(TuiEvent::Error {
                                    context: "websocket receive".into(),
                                    message: e.to_string(),
                                }).await;
                                return Err(e.into());
                            }
                        }
                    }

                    Some(action) = action_rx.recv() => {
                        if matches!(action, TuiAction::Disconnect | TuiAction::Shutdown) {
                            self.send_action(&mut conn, action).await?;
                            let _ = event_tx.send(TuiEvent::ShutdownRequested).await;
                            break;
                        }

                        self.send_action(&mut conn, action).await?;
                    }
            }
        }

        if let Err(error) = conn.close().await {
            debug!(error = %error, "Close error (non-fatal)");
        }

        Ok(())
    }

    /// Converts a TUI action into a websocket message and sends it to the gateway.
    ///
    /// # Arguments
    ///
    /// * `conn` - The active websocket connection used to send the message.
    /// * `action` - The action requested by the terminal UI.
    async fn send_action(&self, conn: &mut NexoConnection, action: TuiAction) -> Result {
        let message = action.into_gateway_message(NexoClient::User(self.config.clone()));
        let frame = Frame::new(message)?;
        conn.send_frame(&frame).await?;
        Ok(())
    }

    /// Handle an incoming Frame from the gateway.
    ///
    /// The loop does not process the next frame until this function returns,
    /// so any long-running operations should be offloaded to a separate task.
    ///
    /// The obvious candidates for offloading are inference and tool call operations, which
    /// can take a long time to complete.
    ///
    /// # Arguments
    ///
    /// * `frame` - The incoming websocket frame received from the gateway.
    /// * `event_tx` - The sender used to publish normalized TUI events.
    async fn handle_frame(&self, frame: Frame, event_tx: &Sender<TuiEvent>) -> Result {
        let (frame_id, payload) = frame.into_parts::<GatewayToUserMessage>()?;
        let message_type: &'static str = (&payload).into();
        info!(frame_id = ?frame_id, message_type = message_type, "Received frame");

        // Guidelines for handling incoming messages:
        //
        // - Messages that are informational only, we can leverage the response.result() helper to log the outcome and return early.
        // - Messages that do not need a reply and can be handled 'immediately' can be handled inline.
        // - Messages that require a response and are long-running should be offloaded to a separate task. Make sure to first send
        //   an Accepted response back to the gateway before offloading the work to a task. This will ensure the gateway knows the request is
        //   being processed and knows to expect follow up events.
        let event = TuiEvent::from_gateway_message(payload)?;
        event_tx.send(event).await.map_err(crate::Error::from)?;

        Ok(())
    }
}
