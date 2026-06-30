//! Gateway runtime, server, and run orchestration for the nexo workspace.
//!
//! The crate centers on four main areas:
//! - `agent` for session state, scheduling, and tool execution loops.
//! - `memory` for persistent storage helpers used by the gateway.
//! - `server` for connection handling, routing, and shared runtime state.

/// Agent runtime, session state, queueing, and tool orchestration.
pub mod agent;
/// Persistent storage helpers for gateway memory features.
pub mod memory;
/// WebSocket server, routing, and shared gateway state.
pub mod server;
/// Test helpers shared with workspace integration tests.
#[doc(hidden)]
pub mod testing;
/// Tool registration and execution helpers shared across gateway subsystems.
pub mod tools;
