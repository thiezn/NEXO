use serde::{Deserialize, Serialize};

use super::ToolExecutionConstraints;

#[cfg(feature = "sqlx")]
use sqlx::sqlite::SqliteRow;
#[cfg(feature = "sqlx")]
use sqlx::{FromRow, Row};
#[cfg(feature = "sqlx")]
use std::io;

/// Declares a tool that may be exposed to a model for tool calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema, Hash, Eq)]
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
}

#[cfg(feature = "sqlx")]
impl<'r> FromRow<'r, SqliteRow> for ToolDefinition {
    fn from_row(row: &SqliteRow) -> Result<Self, sqlx::Error> {
        let parameters_json: String = row.try_get("parameters_json")?;
        let execution_json: String = row.try_get("execution_constraints_json")?;

        Ok(Self {
            name: row.try_get("name")?,
            description: row.try_get("description")?,
            parameters: serde_json::from_str(&parameters_json).map_err(|error| {
                sqlx::Error::Decode(Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid parameters_json '{}': {}", parameters_json, error),
                )))
            })?,
            contract_version: row.try_get("contract_version")?,
            execution: serde_json::from_str::<ToolExecutionConstraints>(&execution_json).map_err(|error| {
                sqlx::Error::Decode(Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid execution_constraints_json '{}': {}", execution_json, error),
                )))
            })?,
        })
    }
}
