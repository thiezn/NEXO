//! Library entry points for the `nexo-node` runtime and supporting modules.

/// The `engine` module contains the implementation of the inference engine used by the nexo user.
pub mod engine;
pub use engine::NexoUser;

/// The `tui` module provides the terminal user interface for the nexo user.
pub mod tui;
pub use tui::{ConnectionStatus, NexoUserState, TuiAction, TuiController, TuiEvent};

/// The `error` module defines error types and result handling for the nexo user.
pub mod error;
pub use error::{Error, Result};
