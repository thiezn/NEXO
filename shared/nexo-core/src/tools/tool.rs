use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::tools::ToolExecutionConstraints;
use crate::tools::{ToolCall, ToolDefinition, ToolResult};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

/// A trait representing an executable tool with metadata and execution constraints.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The type of arguments this tool accepts
    type Args: DeserializeOwned + JsonSchema;

    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Generates the JSON schema for tool arguments
    fn parameters(&self) -> serde_json::Value {
        // This code is the full schema definiton:
        // let schema = schemars::schema_for!(T);
        // serde_json::to_value(schema).expect("valid schema")

        // we don't need the full schema spec so we'll do the following.
        // Note that this doesn't support enums or nested stuff.
        let schema = schemars::schema_for!(Self::Args);
        let mut value = serde_json::to_value(schema).unwrap();

        let obj = value.as_object_mut().expect("schema is object");

        serde_json::json!({
            "type": obj.get("type").cloned().unwrap_or(serde_json::json!("object")),
            "properties": obj.get("properties").cloned().unwrap_or_default(),
            "required": obj.get("required").cloned().unwrap_or_default(),
        })
    }

    /// Optional contract version for compatibility management.
    fn contract_version(&self) -> Option<&str> {
        None
    }

    /// Helper function to generate arguments from a ToolCall.
    fn parse_args(&self, call: &ToolCall) -> Result<Self::Args> {
        serde_json::from_value(call.arguments.clone()).map_err(|e| Error::InvalidRequest {
            message: format!("invalid arguments: {}", e),
        })
    }

    /// Execution-time orchestration constraints.
    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_read_only()
    }

    /// Full tool definition for LLM registration
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
            contract_version: self.contract_version().map(|s| s.to_string()),
            execution: self.execution_constraints(),
        }
    }

    /// Execute the tool with the given call and return the result.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
}
