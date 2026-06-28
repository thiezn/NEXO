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
}
