//! Library entry points for the `nexo-node` runtime and supporting modules.

/// The `config` module defines the configuration structure for the nexo node.
pub mod config;
/// The `engine` module contains the implementation of the inference engine used by the nexo node.
pub mod engine;
/// The `error` module defines error types and result handling for the nexo node.
pub mod error;

pub use config::NexoNodeConfig;
pub use engine::NexoNode;
pub use error::{Error, Result};
