use crate::Result;
use nexo_core::FrameId;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Top-level WebSocket frame envelope.
///
/// This struct represents the outermost layer of a WebSocket message, containing
/// a unique identifier and a payload.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
pub struct Frame {
    /// Unique identifier for the frame.
    pub id: FrameId,

    /// Frame payload object.
    pub payload: serde_json::Value,
}

impl Frame {
    /// Generate a new Frame with a unique identifier and the given payload.
    pub fn new(payload: impl Serialize) -> Result<Self> {
        Ok(Self {
            id: FrameId::new(),
            payload: serde_json::to_value(payload)?,
        })
    }

    /// Generate a new Frame with the given identifier and payload.
    pub fn with_id(id: FrameId, payload: impl Serialize) -> Result<Self> {
        Ok(Self {
            id,
            payload: serde_json::to_value(payload)?,
        })
    }

    /// Consume the frame and deserialize its payload into the requested type.
    pub fn into_parts<T>(self) -> Result<(FrameId, T)>
    where
        T: DeserializeOwned,
    {
        Ok((self.id, serde_json::from_value(self.payload)?))
    }

    /// Get the frame's unique identifier.
    pub fn id(&self) -> FrameId {
        self.id.clone()
    }

    /// Get a reference to the frame's payload.
    pub fn payload(&self) -> &serde_json::Value {
        &self.payload
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::UserToGatewayMessage;
    use crate::{GatewayToUserMessage, NexoResponse};
    use nexo_core::{ClientInfo, DeviceInfo, NexoState, User, UserProperties};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct TestPayload {
        value: String,
    }

    #[test]
    fn into_parts_deserializes_payload() {
        let frame = Frame::new(TestPayload {
            value: "hello".into(),
        })
        .unwrap();
        let expected_id = frame.id();

        let (frame_id, payload): (FrameId, TestPayload) = frame.into_parts().unwrap();

        assert_eq!(frame_id, expected_id);
        assert_eq!(
            payload,
            TestPayload {
                value: "hello".into()
            }
        );
    }

    #[test]
    fn into_parts_round_trips_user_get_state_message() {
        let frame = Frame::new(UserToGatewayMessage::GetState).unwrap();
        let expected_id = frame.id();

        let (frame_id, payload): (FrameId, UserToGatewayMessage) = frame.into_parts().unwrap();

        assert_eq!(frame_id, expected_id);
        assert!(matches!(payload, UserToGatewayMessage::GetState));
    }

    #[test]
    fn websocket_json_round_trips_user_get_state_message() {
        let outbound = Frame::new(UserToGatewayMessage::GetState).unwrap();
        let expected_id = outbound.id();

        let json = serde_json::to_string(&outbound).unwrap();
        let inbound: Frame = serde_json::from_str(&json).unwrap();
        let (frame_id, payload): (FrameId, UserToGatewayMessage) = inbound.into_parts().unwrap();

        assert_eq!(frame_id, expected_id);
        assert!(matches!(payload, UserToGatewayMessage::GetState));
    }

    #[test]
    fn websocket_json_round_trips_non_empty_get_state_response() {
        let mut state = NexoState::new();
        let user = User::from_properties(&UserProperties::new(
            ClientInfo::new("schema-test-user"),
            DeviceInfo::default(),
            "token",
        ));
        state.add_user(user).unwrap();

        let outbound = Frame::new(GatewayToUserMessage::GetState(NexoResponse::Completed {
            operation_id: nexo_core::OperationId::new(),
            result: state,
        }))
        .unwrap();

        let json = serde_json::to_string(&outbound).unwrap();
        let inbound: Frame = serde_json::from_str(&json).unwrap();
        let (_, payload): (FrameId, GatewayToUserMessage) = inbound.into_parts().unwrap();

        let GatewayToUserMessage::GetState(NexoResponse::Completed { result, .. }) = payload else {
            panic!("expected get_state response")
        };
        assert_eq!(result.user_count(), 1);
        assert_eq!(result.node_count(), 0);
    }
}
