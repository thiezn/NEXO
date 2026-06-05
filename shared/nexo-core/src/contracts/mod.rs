//! Cross-crate service contracts implemented by runtime crates.

/// The model registry and selection contract.
pub mod model_registry;
/// The asynchronous tool execution contract.
pub mod tool_executor;

pub use model_registry::ModelRegistry;
pub use tool_executor::ToolExecutor;
