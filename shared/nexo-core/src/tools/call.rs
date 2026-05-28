use serde::{Deserialize, Serialize};

use crate::ids::ToolCallId;

/// A concrete tool call emitted by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct ToolCall {
    /// The unique identifier assigned to the tool call.
    pub id: ToolCallId,

    /// The zero-based order of the call within the assistant response.
    pub index: usize,

    /// The selected tool name.
    pub name: String,

    /// The structured JSON arguments for the tool call.
    pub arguments: serde_json::Value,
}

/// A partial streamed update for a tool call under construction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct ToolCallDelta {
    /// The zero-based order of the call within the response.
    pub index: usize,

    /// The tool call identifier, if known at this point in the stream.
    pub id: Option<ToolCallId>,

    /// The tool name fragment or final name for the call.
    pub name: Option<String>,

    /// The incremental JSON argument text received so far.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_delta: Option<String>,
}
