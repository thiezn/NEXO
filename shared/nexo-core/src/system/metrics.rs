use serde::{Deserialize, Serialize};

/// Metrics related to a Nexo Node, such as CPU usage, memory usage, and other performance indicators.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct NexoNodeMetrics {}
