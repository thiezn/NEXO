use super::{InferenceEvent, NexoEvent, NexoResponse};
use nexo_core::{NexoState, Session, Sessions};
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// The messages that can be sent from a gateway to a user.
#[derive(Debug, IntoStaticStr, Serialize, Deserialize)]
pub enum GatewayToUserMessage {
    /// A NexoResponse::Completed to a user request to indicate the Connect was completed
    /// synchronously and the user is now connected to the gateway.
    Connect(NexoResponse),

    /// A NexoResponse::Completed to a user request to indicate the Disconnect was completed
    /// synchronously and the user is now disconnected from the gateway.
    ///
    /// In the future this could change to a NexoResponse::Accepted to indicate the gateway is
    /// processing the disconnect gracefully and the user should wait for a DisconnectCompleted event
    /// to be sent before the user can consider itself fully disconnected.
    Disconnect(NexoResponse),

    /// A NexoResponse::Completed to a user request with the current state of the Nexo system,
    /// including the available models, nodes, and other relevant information.
    GetState(NexoResponse<NexoState>),

    /// A response to a user request that was accepted for asynchronous processing.
    StartInferenceRun(NexoResponse),

    /// An event emitted for an inference request that was accepted for asynchronous processing.
    StartInferenceRunEvent(NexoEvent<InferenceEvent>),

    /// Reply to a CancelRequest to indicate the cancel request. Depending on the action
    /// thats being cancelled, this could be a NexoResponse::Completed or NexoResponse::Accepted.
    Cancel(NexoResponse),

    /// A response to a user request to list all active sessions.
    ListSessions(NexoResponse<Sessions>),

    /// A response to a user request to get a specific session.
    GetSession(NexoResponse<Session>),

    /// A response to a user request to clear a specific session.
    ClearSession(NexoResponse),
}
