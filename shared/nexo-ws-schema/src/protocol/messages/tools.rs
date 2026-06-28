use nexo_core::{OperationId, ToolCallId, ToolResult};
use serde::{Deserialize, Serialize};

/// The events that can be emitted related to an operation started by a ExecuteTool request.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecuteToolEvent {
    /// The execute tool request has been accepted for asynchronous processing.
    Started {
        /// The operation_id associated with the initial execute request
        operation_id: OperationId,

        /// The original tool_call_id
        tool_call_id: ToolCallId,
    },

    /// The execute tool request has completed successfully.
    Completed {
        /// The operation_id associated with the initial execute request
        operation_id: OperationId,

        /// The original tool_call_id
        tool_call_id: ToolCallId,

        /// The result of the tool execution
        result: ToolResult,
    },

    /// The execute tool request has failed with an error.
    Failed {
        /// The operation_id associated with the initial execute request
        operation_id: OperationId,

        /// The original tool_call_id
        tool_call_id: ToolCallId,

        /// The error message describing the failure.
        error: String,
    },
}
