//! Round-based run execution loop and supporting helpers.

mod context_manager;
mod engine;
mod events;
mod inference;
pub(crate) mod router;

pub(crate) use engine::run_existing;
pub use engine::start_run;
