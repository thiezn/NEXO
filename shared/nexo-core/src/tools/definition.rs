use serde::{Deserialize, Serialize};

use super::ToolExecutionConstraints;

/// Declares a tool that may be exposed to a model for tool calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub struct ToolDefinition {
    /// The unique tool name used in model-facing schemas.
    pub name: String,

    /// The human-readable tool description.
    pub description: String,

    /// The JSON Schema object describing tool parameters.
    pub parameters: serde_json::Value,

    /// An optional contract version for the tool definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_version: Option<String>,

    /// Execution-time orchestration constraints.
    #[serde(default)]
    pub execution: ToolExecutionConstraints,
    // /// Additional tool metadata for higher-level consumers.
    // #[serde(default)]
    // pub metadata: MetadataMap,
}
