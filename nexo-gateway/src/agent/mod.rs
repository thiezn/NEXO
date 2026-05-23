//! Agent orchestration, persistence, and prompt-building modules.

/// Conversation context loading and stored prompt assets for agent runs.
pub mod context;
/// Cron-style scheduled agent jobs.
pub mod cron;
/// Capability lock management for tool execution.
pub mod locks;
/// Round-based execution loop for agent runs.
pub mod r#loop;
/// Queue management for deferred runs.
pub mod queue;
mod runtime;
/// Session, transcript, and run persistence helpers.
pub mod session;

pub use runtime::{AgentCommand, AgentHandle};
