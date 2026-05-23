//! Coordinator orchestration, lifecycle operations, and factory wiring.

mod core;
pub mod factory;
pub mod load;
pub mod unload;

pub use core::{Coordinator, ModelSlot};
