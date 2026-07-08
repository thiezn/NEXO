use serde::{Deserialize, Serialize};

/// Declares the side-effect profile of a tool.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolSideEffectLevel {
    /// The tool is read-only and may be scheduled more aggressively.
    ReadOnly,

    /// The tool may mutate external state.
    #[default]
    SideEffecting,
}

/// Declares whether a tool may run in parallel with other tool calls.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, Hash,
)]
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
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, Hash,
)]
#[serde(rename_all = "snake_case")]
pub struct ToolExecutionConstraints {
    /// The declared side-effect level of the tool.
    pub side_effect_level: ToolSideEffectLevel,

    /// The parallel execution policy of the tool.
    pub parallelism: ToolParallelism,

    /// An optional timeout budget, expressed in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl ToolExecutionConstraints {
    /// Default execution constraints for read-only tools.
    pub fn default_read_only() -> Self {
        Self {
            side_effect_level: ToolSideEffectLevel::ReadOnly,
            parallelism: ToolParallelism::ParallelGlobal,
            timeout_ms: None,
        }
    }

    /// Default execution constraints for side-effecting tools.
    pub fn default_side_effecting() -> Self {
        Self {
            side_effect_level: ToolSideEffectLevel::SideEffecting,
            parallelism: ToolParallelism::Sequential,
            timeout_ms: None,
        }
    }
}
