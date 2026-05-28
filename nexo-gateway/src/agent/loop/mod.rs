//! Round-based run execution loop and supporting helpers.

mod context_manager;
mod engine;
mod events;

pub use engine::{resume_run, start_run};
