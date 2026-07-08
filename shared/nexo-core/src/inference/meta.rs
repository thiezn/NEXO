use serde::{Deserialize, Serialize};

use crate::{InferenceRequest, ModelId, OperationId, RoundId, RunId};

/// Stable identity shared by inference runtime updates and transport events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceMeta {
    /// The operation identifier associated with the inference request.
    pub operation_id: OperationId,

    /// The run identifier associated with the inference request.
    pub run_id: RunId,

    /// The round identifier associated with the inference request.
    pub round_id: RoundId,

    /// The model selected to execute the request.
    pub model_id: ModelId,
}

impl InferenceMeta {
    /// Builds inference metadata after a request has been resolved to a concrete model.
    pub fn from_request(request: &InferenceRequest) -> Self {
        Self {
            operation_id: request.operation_id,
            run_id: request.run_id,
            round_id: request.round_id.clone(),
            model_id: request.model_id,
        }
    }
}
