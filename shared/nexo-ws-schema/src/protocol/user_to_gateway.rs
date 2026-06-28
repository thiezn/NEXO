use super::CancelRequest;
use nexo_core::{InferenceRequest, OperationId, SessionId, UserProperties};
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// The messages that can be sent from a user to the gateway.
#[derive(Debug, IntoStaticStr, Serialize, Deserialize)]
pub enum UserToGatewayMessage {
    /// Connect to the gateway and establish a session.
    ///
    /// TODO: ConnectParams should split into generic connect details and user specific params.
    /// Same for Node connect
    Connect(UserProperties),

    /// Disconnect from the gateway and close the session gracefully.
    Disconnect,

    /// Request the current state of the Nexo system, including the available models,
    /// nodes, and other relevant information.
    ///
    /// NOTE: This replaces all previous status, health, systempresence and tick events.
    /// We rely on WebSocket protocol to keep the connection alive and send pings, so we
    /// don't need to send status events anymore.
    ///
    /// The gateway will periodically send state events to the user so we can
    /// update the user interface with the latest information about the system. This event
    /// can force a state update.
    GetState,

    /// A request to start a new inference run operation with the specified parameters.
    StartInferenceRun(InferenceRequest),

    /// A request to append additional instructions to an ongoing inference run operation.
    ///
    /// TODO: Review the required payload.
    AppendInferenceInstructions {
        /// The unique identifier for the inference operation to which the instructions should be appended.
        operation_id: OperationId,

        /// The additional instructions to be appended to the ongoing inference operation.
        instructions: InferenceRequest,
    },

    /// A request to cancel a previously submitted operation.
    Cancel(CancelRequest),

    /// List all active sessions
    ListSessions,

    /// Get a specific Session
    GetSession {
        /// The unique identifier of the session to retrieve.
        session_id: SessionId,
    },

    /// Clear a specific Session
    ClearSession {
        /// The unique identifier of the session to clear.
        session_id: SessionId,
    },
}
