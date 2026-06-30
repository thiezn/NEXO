use serde::{Deserialize, Serialize};

use crate::common::Timestamp;
use crate::ids::{ModelId, RunId};

use super::{RoundEvent, RunStatus};

/// A point-in-time run status update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RunStatusUpdate {
    /// The run identifier associated with the update.
    pub run_id: RunId,

    /// The new run status.
    pub status: RunStatus,

    /// The selected model, if one has been resolved.
    pub model_id: Option<ModelId>,

    /// The timestamp at which the update was emitted.
    pub timestamp: Timestamp,

    /// An optional human-readable status message.
    pub message: Option<String>,
}

/// An event emitted over the lifecycle of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum RunEvent {
    /// A lifecycle status update.
    Status(RunStatusUpdate),

    /// A round-scoped event emitted within the run.
    Round(RoundEvent),
}
