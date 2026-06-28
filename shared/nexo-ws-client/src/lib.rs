//! Async websocket client helpers for speaking the NEXO gateway protocol.

/// Websocket connection primitives.
pub mod connection;
/// Client error types.
pub mod error;

pub use connection::{NexoConnection, ReadHalf, WriteHalf};
pub use error::{Error, Result};
