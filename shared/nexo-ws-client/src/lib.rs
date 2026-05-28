//! Async websocket client helpers for speaking the NEXO gateway protocol.

/// Websocket connection primitives.
pub mod connection;
/// Client error types.
pub mod error;
/// Gateway handshake helpers.
pub mod handshake;

pub use connection::{NexoConnection, ReadHalf, WriteHalf};
pub use error::{ClientError, Result};
pub use handshake::{default_node_connect_params, default_user_connect_params, perform_handshake};
