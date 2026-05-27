//! Cross-crate service contracts implemented by runtime crates.

/// The generic inference engine contract.
pub mod inference_engine;
/// The model registry and selection contract.
pub mod model_registry;
/// The asynchronous tool execution contract.
pub mod tool_executor;

pub use inference_engine::InferenceEngine;
pub use model_registry::ModelRegistry;
pub use tool_executor::ToolExecutor;
