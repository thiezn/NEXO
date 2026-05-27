use serde::{Deserialize, Serialize};

use crate::ids::ToolCallId;

/// Indicates whether a tool execution succeeded or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    /// The tool completed successfully.
    Success,

    /// The tool completed with an execution failure.
    Failure,
}

/// The payload returned by a tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ToolResultContent {
    /// Plain textual tool output.
    Text(String),

    /// Structured JSON tool output.
    Json(serde_json::Value),
}

/// A tool execution result that can be fed back into a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolResult {
    /// The identifier of the tool call this result satisfies.
    pub tool_call_id: ToolCallId,

    /// The name of the tool that produced the result.
    pub tool_name: String,

    /// The success or failure status of the execution.
    pub status: ToolResultStatus,

    /// The result payload returned by the tool.
    pub content: ToolResultContent,
}
