use serde::{Deserialize, Serialize};

/// The current lifecycle state of a gateway-managed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The run has been accepted but not started.
    Queued,

    /// The run is preparing model state or resolving resources.
    Preparing,

    /// The run is actively generating output.
    Running,

    /// The run is in an explicit thinking phase.
    Thinking,

    /// The run is waiting for tool execution to complete.
    WaitingForTool,

    /// The run completed successfully.
    Completed,

    /// The run failed.
    Failed,

    /// The run was cancelled.
    Cancelled,
}
