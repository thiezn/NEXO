//! WebSocket transport, routing, and shared server-side state.

/// Authentication helpers for the WebSocket upgrade.
pub mod auth;
/// WebSocket request handling and dispatch.
pub mod handler;
/// Shared state for connected peers, routing, and tool catalogs.
pub mod state;
/// Periodic event broadcasting for connected peers.
pub mod ticker;
