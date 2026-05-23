//! Round-based agent execution loop and supporting helpers.

mod engine;
mod events;
mod queue;

pub use engine::{resume_run, start_run};
