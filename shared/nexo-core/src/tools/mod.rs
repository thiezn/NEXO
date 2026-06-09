//! Tool declarations, execution policies, calls, and results.

/// Tool call and tool call delta types.
pub mod call;
/// Tool choice policy types.
pub mod choice;
/// Tool schema and definition types.
pub mod definition;
/// Tool execution policy types.
pub mod execution;
/// Generic tool registry for core tool definitions and executors.
pub mod registry;

/// Tool trait implementation
pub mod tool;

/// Tool result payload types.
pub mod result;

pub use call::{ToolCall, ToolCallDelta};
pub use choice::ToolChoice;
pub use definition::ToolDefinition;
pub use execution::{ToolExecutionConstraints, ToolParallelism, ToolSideEffectLevel};
pub use registry::ToolRegistry;
pub use result::{ToolResult, ToolResultContent, ToolResultStatus};
pub use tool::Tool;
