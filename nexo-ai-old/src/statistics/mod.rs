//! Statistics collection, storage backends, and rendering helpers.

pub mod aggregates;
pub mod backend;
mod collector;
pub mod display;
pub mod metrics;

pub use collector::StatsCollector;
