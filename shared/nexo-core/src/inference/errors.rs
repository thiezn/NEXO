use serde::{Deserialize, Serialize};

use crate::ids::{RequestId, RoundId, RunId};

/// Indicates whether a failed request may be retried safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Retryability {
    /// The failure may be retried.
    Retryable,

    /// The failure should be treated as terminal.
    Fatal,
}

/// A coarse-grained error code for inference failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InferenceErrorCode {
    /// The request payload is invalid.
    InvalidRequest,

    /// The request asked for a feature not supported by the selected model.
    UnsupportedFeature,

    /// The target model is unavailable.
    ModelUnavailable,

    /// A tool call or tool round-trip failed.
    ToolFailure,

    /// The request was cancelled before completion.
    Cancelled,

    /// An internal runtime error occurred.
    Internal,
}

/// A structured failure returned from an inference engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceFailure {
    /// The request identifier associated with the failure, if one exists.
    pub request_id: Option<RequestId>,

    /// The run identifier associated with the failure, if one exists.
    pub run_id: Option<RunId>,

    /// The round identifier associated with the failure, if one exists.
    pub round_id: Option<RoundId>,

    /// The coarse-grained failure code.
    pub code: InferenceErrorCode,

    /// The human-readable failure message.
    pub message: String,

    /// Indicates whether a retry may succeed.
    pub retryability: Retryability,
}
