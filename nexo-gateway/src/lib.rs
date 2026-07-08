//! Gateway runtime, server, and run orchestration for the nexo workspace.
//!
//! The crate centers on four main areas:
//! - `agent` for session state, scheduling, and tool execution loops.
//! - `memory` for persistent storage helpers used by the gateway.
//! - `server` for connection handling, routing, and shared runtime state.

/// CLI parsing and command dispatch for the nexo gateway binary.
pub mod cli;

/// Persistent storage helpers for gateway memory features.
pub mod memory;

/// The main gateway agent, responsible for scheduling and executing tool calls.
pub mod engine;
pub use engine::NexoGateway;

/// Error types and result handling for the nexo gateway.
pub mod error;
pub use error::{Error, Result};

/// The main Nexo Agent. This is the heart of the NEXO system running the loop
pub mod agent;
use agent::NexoAgent;
