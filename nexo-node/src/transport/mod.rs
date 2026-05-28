//! Websocket transport, connection lifecycle, and protocol helpers for `nexo-node`.

mod inference;
mod protocol;
mod runtime;

pub(crate) use protocol::send;
pub use runtime::run_node;
