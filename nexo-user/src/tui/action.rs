use crate::{Error, Result};
use nexo_core::inference::MultiModalPayload;
use nexo_core::{
    ConversationMessage, InferenceIntent, InferenceOperation, ModelCapability, ModelSelection,
    NexoClient, OperationId, ReasoningSettings, SessionId, ToolChoice,
};
use nexo_ws_schema::{CancelRequest, ConnectRequest, DisconnectRequest, UserToGatewayMessage};
use strum::IntoStaticStr;

/// A transport-agnostic action emitted by the terminal UI.
#[derive(Debug, Clone, IntoStaticStr)]
pub enum TuiAction {
    /// Requests that the engine establish or re-establish the user session with the gateway.
    Connect,

    /// Requests that the engine disconnect cleanly from the gateway.
    Disconnect,

    /// Requests that the engine shut down all active runtime components.
    Shutdown,

    /// Requests the latest gateway state snapshot.
    RefreshState,

    /// Requests the current list of active sessions.
    ListSessions,

    /// Requests the full details for a specific session.
    GetSession {
        /// The identifier of the session to retrieve.
        session_id: SessionId,
    },

    /// Requests that a specific session be cleared.
    ClearSession {
        /// The identifier of the session to clear.
        session_id: SessionId,
    },

    /// Requests that a new inference run be started.
    StartInferenceRun {
        /// The fully typed inference request to submit to the gateway.
        request: InferenceIntent,
    },

    /// Requests that a previously submitted operation be cancelled.
    Cancel {
        /// The identifier of the operation to cancel.
        operation_id: OperationId,
    },
}

impl TuiAction {
    /// Converts this TUI action into the corresponding gateway message.
    ///
    /// # Arguments
    ///
    /// * `client` - The fully configured client identity used for connect requests.
    pub fn into_gateway_message(self, client: NexoClient) -> UserToGatewayMessage {
        match self {
            Self::Connect => UserToGatewayMessage::Connect(ConnectRequest::new(client)),
            Self::Disconnect | Self::Shutdown => {
                UserToGatewayMessage::Disconnect(DisconnectRequest::new())
            }
            Self::RefreshState => UserToGatewayMessage::GetState,
            Self::ListSessions => UserToGatewayMessage::ListSessions,
            Self::GetSession { session_id } => UserToGatewayMessage::GetSession { session_id },
            Self::ClearSession { session_id } => UserToGatewayMessage::ClearSession { session_id },
            Self::StartInferenceRun { request } => UserToGatewayMessage::StartInferenceRun(request),
            Self::Cancel { operation_id } => {
                UserToGatewayMessage::Cancel(CancelRequest { operation_id })
            }
        }
    }

    /// Builds the default plain-text multimodal inference action.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user-entered plain-text prompt.
    /// * `session_id` - The session to attach to the request, or a new one if absent.
    pub fn from_plain_text_prompt(prompt: String, session_id: Option<SessionId>) -> Result<Self> {
        let trimmed_prompt = prompt.trim();
        if trimmed_prompt.is_empty() {
            return Err(Error::NexoCore(nexo_core::Error::InvalidRequest {
                message: "plain-text prompt cannot be empty".into(),
            }));
        }

        let request = InferenceIntent {
            operation_id: OperationId::new(),
            session_id: session_id.unwrap_or_else(SessionId::new),
            model_selection: ModelSelection::Capabilities(vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
                ModelCapability::Reasoning,
                ModelCapability::Streaming,
            ]),
            operation: InferenceOperation::MultiModal(MultiModalPayload::new_round(
                vec![ConversationMessage::new_text(trimmed_prompt)],
                Vec::new(),
                ToolChoice::Automatic,
                ReasoningSettings::default(),
            )),
        };

        Ok(Self::StartInferenceRun { request })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexo_core::{ClientInfo, DeviceInfo, UserProperties};

    fn make_client() -> NexoClient {
        NexoClient::User(UserProperties::new(
            ClientInfo::new(env!("CARGO_PKG_VERSION")),
            DeviceInfo::default(),
            nexo_ws_schema::AUTH_TOKEN,
        ))
    }

    #[test]
    fn maps_refresh_state_action_to_get_state_message() {
        let message = TuiAction::RefreshState.into_gateway_message(make_client());

        assert!(matches!(message, UserToGatewayMessage::GetState));
    }

    #[test]
    fn builds_default_plain_text_prompt_with_capability_selection() {
        let action = TuiAction::from_plain_text_prompt("summarize this".into(), None)
            .expect("plain-text prompt should build a default inference action");

        let TuiAction::StartInferenceRun { request } = action else {
            panic!("expected StartInferenceRun action");
        };

        assert!(matches!(
            request.model_selection,
            ModelSelection::Capabilities(ref capabilities)
            if capabilities == &vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
                ModelCapability::Reasoning,
                ModelCapability::Streaming,
            ]
        ));
    }
}
