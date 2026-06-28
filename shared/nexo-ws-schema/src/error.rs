use nexo_core::OperationId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

/// Result alias for ws-schema operations.
pub type Result<T = ()> = std::result::Result<T, Error>;

/// Serializable wrapper for `serde_json::Error` that preserves only its display text.
#[derive(Debug, Clone, ThisError, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[error("{0}")]
#[serde(transparent)]
pub struct SerializationErrorMessage(String);

impl From<serde_json::Error> for SerializationErrorMessage {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

/// Protocol-level errors surfaced by the ws-schema transport layer.
#[derive(Debug, ThisError, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum Error {
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

    /// Expected a synchronous completed response
    #[error(
        "Expected synchronous completed response, but got: {error} (operation_id: {operation_id})"
    )]
    ExpectedCompletedResponse {
        /// Operation ID of the request that was expected to be completed.
        operation_id: OperationId,

        /// Error payload from the remote side.
        error: String,
    },

    /// Expected an asynchronous accepted response
    #[error(
        "Expected asynchronous accepted response, but got: {error} (operation_id: {operation_id})"
    )]
    ExpectedAcceptedResponse {
        /// Operation ID of the request that was expected to be accepted.
        operation_id: OperationId,

        /// Error payload from the remote side.
        error: String,
    },

    /// Failed response from the remote side.
    #[error("Response failed: {error} (operation_id: {operation_id})")]
    ResponseFailed {
        /// Operation ID of the failed request.
        operation_id: OperationId,
        /// Error payload from the remote side.
        error: String,
    },

    /// JSON serialization/deserialization failure.
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationErrorMessage),

    /// Message processing failure.
    #[error("Message error: {0}")]
    Message(String),

    /// Operation timed out.
    #[error("Timeout: {0}")]
    Timeout(String),
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error.into())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn error_display_protocol() {
        let e = Error::Protocol("bad frame".into());
        assert_eq!(e.to_string(), "Protocol error: bad frame");
    }

    #[test]
    fn error_display_protocol_mismatch() {
        let e = Error::ProtocolMismatch {
            min: 1,
            max: 2,
            server: 3,
        };
        assert!(e.to_string().contains("client [1..2]"));
        assert!(e.to_string().contains("server 3"));
    }

    #[test]
    fn error_serialization_roundtrip() {
        let error = Error::Auth("invalid token".into());
        let json = serde_json::to_string(&error).unwrap();
        let decoded: Error = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded, Error::Auth(message) if message == "invalid token"));
    }
}
