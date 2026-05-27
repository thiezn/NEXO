use crate::error::Result;
use crate::tools::{ToolCall, ToolResult};

/// A service capable of executing tool calls outside the model runtime.
#[allow(async_fn_in_trait)]
pub trait ToolExecutor: Send + Sync {
    /// Executes a tool call and returns the resulting tool output.
    ///
    /// # Arguments
    ///
    /// * `call` - The tool call to execute.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
}
