use crate::Result;
use nexo_core::{NexoState, OperationId, Session, Sessions};
use nexo_ws_schema::{GatewayToUserMessage, InferenceRunEvent, NexoResponse};
use strum::IntoStaticStr;

/// A normalized event sent from the engine to the terminal UI.
#[derive(Debug, IntoStaticStr)]
pub enum TuiEvent {
    /// Indicates that the websocket connection and handshake are active.
    Connected,

    /// Indicates that the gateway connection has been closed or disconnected.
    Disconnected,

    /// Indicates that the application is beginning a graceful shutdown sequence.
    ShutdownRequested,

    /// Publishes the latest gateway state snapshot.
    StateUpdated {
        /// The current gateway state payload.
        state: NexoState,
    },

    /// Publishes the latest session list snapshot.
    SessionsListed {
        /// The sessions returned by the gateway.
        sessions: Sessions,
    },

    /// Publishes the full details for a specific session.
    SessionLoaded {
        /// The session returned by the gateway.
        session: Session,
    },

    /// Indicates that an operation was accepted for asynchronous processing.
    OperationAccepted {
        /// The operation identifier correlated to the accepted request.
        operation_id: OperationId,

        /// A short context label describing the operation.
        context: String,
    },

    /// Indicates that an operation completed synchronously.
    OperationCompleted {
        /// The operation identifier correlated to the completed request.
        operation_id: OperationId,

        /// A short context label describing the operation.
        context: String,
    },

    /// Indicates that an operation failed.
    OperationFailed {
        /// The operation identifier correlated to the failed request.
        operation_id: OperationId,

        /// A short context label describing the operation.
        context: String,

        /// The human-readable failure message.
        message: String,
    },

    /// Publishes a streaming inference run event.
    InferenceRunEvent {
        /// The streaming inference event received from the gateway.
        event: nexo_ws_schema::NexoEvent<InferenceRunEvent>,
    },

    /// Publishes a user-visible error unrelated to a typed gateway response.
    Error {
        /// A short label for the failing subsystem or operation.
        context: String,

        /// The human-readable error message.
        message: String,
    },
}

impl TuiEvent {
    /// Builds a normalized TUI event from a gateway message payload.
    ///
    /// # Arguments
    ///
    /// * `payload` - The typed gateway message received from the websocket transport.
    pub fn from_gateway_message(payload: GatewayToUserMessage) -> Result<Self> {
        let event = match payload {
            GatewayToUserMessage::Connect(response) => Self::from_response("connect", response),
            GatewayToUserMessage::Disconnect(response) => {
                response.result()?;
                Self::Disconnected
            }
            GatewayToUserMessage::GetState(response) => {
                Self::from_data_response(response, |state| Self::StateUpdated { state })
            }
            GatewayToUserMessage::ListSessions(response) => {
                Self::from_data_response(response, |sessions| Self::SessionsListed { sessions })
            }
            GatewayToUserMessage::GetSession(response) => {
                Self::from_data_response(response, |session| Self::SessionLoaded { session })
            }
            GatewayToUserMessage::ClearSession(response) => {
                Self::from_response("clear_session", response)
            }
            GatewayToUserMessage::StartInferenceRun(response) => {
                Self::from_response("start_inference_run", response)
            }
            GatewayToUserMessage::StartInferenceRunEvent(event) => Self::InferenceRunEvent { event },
            GatewayToUserMessage::Cancel(response) => Self::from_response("cancel", response),
        };

        Ok(event)
    }

    /// Builds a TUI event from a response without payload data.
    ///
    /// # Arguments
    ///
    /// * `context` - A short label describing the operation that produced the response.
    /// * `response` - The response received from the gateway.
    fn from_response(context: &str, response: NexoResponse) -> Self {
        match response {
            NexoResponse::Completed { operation_id, .. } => Self::OperationCompleted {
                operation_id,
                context: context.into(),
            },
            NexoResponse::Accepted { operation_id } => Self::OperationAccepted {
                operation_id,
                context: context.into(),
            },
            NexoResponse::Failed {
                operation_id,
                error,
            } => Self::OperationFailed {
                operation_id,
                context: context.into(),
                message: error.to_string(),
            },
        }
    }

    /// Builds a TUI event from a response carrying result data.
    ///
    /// # Arguments
    ///
    /// * `response` - The response containing typed gateway result data.
    /// * `map_completed` - The function used to convert a completed response payload into a TUI event.
    fn from_data_response<T>(response: NexoResponse<T>, map_completed: impl FnOnce(T) -> Self) -> Self {
        match response {
            NexoResponse::Completed { result, .. } => map_completed(result),
            NexoResponse::Accepted { operation_id } => Self::OperationAccepted {
                operation_id,
                context: "gateway_operation".into(),
            },
            NexoResponse::Failed {
                operation_id,
                error,
            } => Self::OperationFailed {
                operation_id,
                context: "gateway_operation".into(),
                message: error.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_list_sessions_response_to_tui_event() {
        let payload = GatewayToUserMessage::ListSessions(NexoResponse::Completed {
            operation_id: OperationId::new(),
            result: Vec::new(),
        });

        let event = TuiEvent::from_gateway_message(payload)
            .expect("payload should map to a TUI event");

        assert!(matches!(event, TuiEvent::SessionsListed { sessions } if sessions.is_empty()));
    }
}
