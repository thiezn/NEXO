//! Run orchestration, persistence, and prompt-building modules.

/// Transcript loading and stored prompt assets for runs.
pub mod context;
/// Cron-style scheduled run jobs.
pub mod cron;
/// Capability lock management for tool execution.
pub mod locks;
/// Round-based execution loop for runs.
pub mod r#loop;
/// Queue management for deferred runs.
pub mod queue;
mod runtime;
/// Session, transcript, and run persistence helpers.
pub mod session;

pub use runtime::{RunCommand, RunHandle};
