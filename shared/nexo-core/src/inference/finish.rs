use serde::{Deserialize, Serialize};

/// The reason a generation operation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model completed normally.
    Completed,

    /// A configured stop sequence was encountered.
    StopSequence,

    /// The response reached the maximum output token budget.
    MaxTokens,

    /// The model ended generation because it emitted tool calls.
    ToolCalls,

    /// The request was cancelled by the caller or runtime.
    Cancelled,

    /// The response was filtered or withheld by policy.
    ContentFiltered,
}
