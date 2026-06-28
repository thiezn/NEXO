use serde::{Deserialize, Serialize};

/// The state of the Nexo System.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct NexoState {}
