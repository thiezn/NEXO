use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Protocol-level errors surfaced by the ws-schema transport layer.
#[derive(Debug, Error)]
pub enum WsError {
    /// Generic protocol framing or semantic error.
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Request/response payload validation error.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Underlying connection failure.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Handshake negotiation failed.
    #[error("Handshake failed: {0}")]
    Handshake(String),

    /// Authentication or authorization failure.
    #[error("Auth failed: {0}")]
    Auth(String),

    /// Client/server protocol range mismatch.
    #[error("Unsupported protocol version: client [{min}..{max}], server {server}")]
    ProtocolMismatch {
        /// Minimum protocol version supported by the client.
        min: u32,
        /// Maximum protocol version supported by the client.
        max: u32,
        /// Protocol version supported by the server.
        server: u32,
    },

    /// Requested method was not recognized.
    #[error("Method not found: {0}")]
    MethodNotFound(String),

    /// JSON serialization/deserialization failure.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Message processing failure.
    #[error("Message error: {0}")]
    Message(String),

    /// Operation timed out.
    #[error("Timeout: {0}")]
    Timeout(String),
}

/// Result alias for ws-schema operations.
pub type Result<T = ()> = std::result::Result<T, WsError>;

/// Wire-format error payload included in error responses.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ErrorPayload {
    /// Stable machine-readable error code.
    pub code: String,
    /// Human-readable error message.
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
