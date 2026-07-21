use crate::{OperationId, ToolCallId, ToolResult};
use serde::{Deserialize, Serialize};

/// An event emitted while a node executes a tool call.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecuteToolEvent {
    /// The node started executing the tool call.
    Started {
        /// Operation associated with the tool execution.
        operation_id: OperationId,
        /// Tool call being executed.
        tool_call_id: ToolCallId,
    },
    /// The node completed the tool call successfully.
    Completed {
        /// Operation associated with the tool execution.
        operation_id: OperationId,
        /// Tool call that completed.
        tool_call_id: ToolCallId,
        /// Result produced by the tool.
        result: ToolResult,
    },
    /// The node failed to execute the tool call.
    Failed {
        /// Operation associated with the tool execution.
        operation_id: OperationId,
        /// Tool call that failed.
        tool_call_id: ToolCallId,
        /// Human-readable failure detail.
        error: String,
    },
}
