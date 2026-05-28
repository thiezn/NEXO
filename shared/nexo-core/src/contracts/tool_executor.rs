use crate::error::Result;
use crate::tools::{ToolCall, ToolResult};
use std::future::Future;

/// A service capable of executing tool calls outside the model runtime.
pub trait ToolExecutor: Send + Sync {
    /// Executes a tool call and returns the resulting tool output.
    ///
    /// # Arguments
    ///
    /// * `call` - The tool call to execute.
    fn execute(&self, call: ToolCall) -> impl Future<Output = Result<ToolResult>> + Send;
}
