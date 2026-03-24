use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WsError {
    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Handshake failed: {0}")]
    Handshake(String),

    #[error("Auth failed: {0}")]
    Auth(String),

    #[error("Unsupported protocol version: client [{min}..{max}], server {server}")]
    ProtocolMismatch { min: u32, max: u32, server: u32 },

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Message error: {0}")]
    Message(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

pub type Result<T = ()> = std::result::Result<T, WsError>;

/// Wire-format error payload included in error responses.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
}

impl From<&WsError> for ErrorPayload {
    fn from(err: &WsError) -> Self {
        let code = match err {
            WsError::Protocol(_) => "protocol_error",
            WsError::Validation(_) => "validation_error",
            WsError::Connection(_) => "connection_error",
            WsError::Handshake(_) => "handshake_failed",
            WsError::Auth(_) => "auth_failed",
            WsError::ProtocolMismatch { .. } => "protocol_mismatch",
            WsError::MethodNotFound(_) => "method_not_found",
            WsError::Serialization(_) => "serialization_error",
            WsError::Message(_) => "message_error",
            WsError::Timeout(_) => "timeout",
        };
        ErrorPayload {
            code: code.to_string(),
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn error_display_protocol() {
        let e = WsError::Protocol("bad frame".into());
        assert_eq!(e.to_string(), "Protocol error: bad frame");
    }

    #[test]
    fn error_display_protocol_mismatch() {
        let e = WsError::ProtocolMismatch {
            min: 1,
            max: 2,
            server: 3,
        };
        assert!(e.to_string().contains("client [1..2]"));
        assert!(e.to_string().contains("server 3"));
    }

    #[test]
    fn error_payload_conversion() {
        let e = WsError::Auth("invalid token".into());
        let payload = ErrorPayload::from(&e);
        assert_eq!(payload.code, "auth_failed");
        assert!(payload.message.contains("invalid token"));
    }

    #[test]
    fn error_payload_serialization_roundtrip() {
        let payload = ErrorPayload {
            code: "test_error".into(),
            message: "something broke".into(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: ErrorPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }
}
