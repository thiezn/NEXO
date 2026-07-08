use crate::{OperationId, SessionId};
use serde::{Deserialize, Serialize};

/// A unified request struct for all supported inference operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct CompactionRequest {
    /// The unique identifier for the inference request.
    pub operation_id: OperationId,

    /// The session identifier for the request
    pub session_id: SessionId,

    /// Optional guide for the compaction request.
    pub instructions: Option<String>,
}
