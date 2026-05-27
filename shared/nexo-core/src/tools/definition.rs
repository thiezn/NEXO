use serde::{Deserialize, Serialize};

use crate::common::MetadataMap;

use super::ToolExecutionConstraints;

/// Declares a tool that may be exposed to a model for tool calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolDefinition {
    /// The unique tool name used in model-facing schemas.
    pub name: String,

    /// The human-readable tool description.
    pub description: String,

    /// The JSON Schema object describing tool parameters.
    pub parameters: serde_json::Value,

    /// An optional contract version for the tool definition.
    pub contract_version: Option<String>,

    /// Execution-time orchestration constraints.
    pub execution: ToolExecutionConstraints,

    /// Additional tool metadata for higher-level consumers.
    pub metadata: MetadataMap,
}
