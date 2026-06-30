use serde::{Deserialize, Serialize};

use crate::common::Timestamp;
use crate::ids::{ModelId, OperationId, RoundId, RunId};
use crate::inference::{InferenceUpdate, TokenUsage};

/// The current lifecycle state of a single round within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RoundStatus {
    /// The round has been accepted but not started.
    Queued,

    /// The round is preparing model state or assembling the prompt.
    Preparing,

    /// The round is actively generating output.
    Running,

    /// The round is in an explicit thinking phase.
    Thinking,

    /// The round is waiting for tool execution to complete.
    WaitingForTool,

    /// The round completed successfully.
    Completed,

    /// The round failed.
    Failed,

    /// The round was cancelled.
    Cancelled,
}

/// A point-in-time status update for a single round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RoundStatusUpdate {
    /// The run identifier that owns the round.
    pub run_id: RunId,

    /// The unique round identifier.
    pub round_id: RoundId,

    /// The operation identifier associated with the round, if one has been assigned.
    pub operation_id: Option<OperationId>,

    /// The new round status.
    pub status: RoundStatus,

    /// The selected model, if one has been resolved.
    pub model_id: Option<ModelId>,

    /// The timestamp at which the update was emitted.
    pub timestamp: Timestamp,

    /// An optional human-readable status message.
    pub message: Option<String>,
}

/// An event emitted over the lifecycle of a single round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum RoundEvent {
    /// A lifecycle status update.
    Status(RoundStatusUpdate),

    /// An inference update emitted for the round.
    Inference(InferenceUpdate),
}

/// A compact summary of a completed or in-flight round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RoundSummary {
    /// The owning run identifier.
    pub run_id: RunId,

    /// The unique round identifier.
    pub round_id: RoundId,

    /// The operation identifier associated with the round, if one has been assigned.
    pub operation_id: Option<OperationId>,

    /// The current round status.
    pub status: RoundStatus,

    /// The selected model, if one has been resolved.
    pub model_id: Option<ModelId>,

    /// The timestamp at which the round started, if known.
    pub started_at: Option<Timestamp>,

    /// The timestamp at which the round ended, if known.
    pub completed_at: Option<Timestamp>,

    /// Token usage recorded for the round, if available.
    pub usage: Option<TokenUsage>,
}
