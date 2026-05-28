use serde::{Deserialize, Serialize};

/// Declares the side-effect profile of a tool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ToolSideEffectLevel {
    /// The tool is read-only and may be scheduled more aggressively.
    ReadOnly,

    /// The tool may mutate external state.
    #[default]
    SideEffecting,
}

/// Declares whether a tool may run in parallel with other tool calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ToolParallelism {
    /// The tool must run sequentially.
    #[default]
    Sequential,

    /// The tool may run concurrently across different nodes, but not on the same node.
    ParallelPerNode,

    /// The tool may run concurrently across the full orchestrator.
    ParallelGlobal,
}

/// Execution constraints used by orchestrators when scheduling a tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionConstraints {
    /// The declared side-effect level of the tool.
    pub side_effect_level: ToolSideEffectLevel,

    /// The parallel execution policy of the tool.
    pub parallelism: ToolParallelism,

    /// An optional timeout budget, expressed in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}
