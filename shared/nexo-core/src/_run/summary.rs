use serde::{Deserialize, Serialize};

use crate::common::Timestamp;
use crate::ids::{ModelId, RunId};
use crate::inference::TokenUsage;

use super::{RoundSummary, RunStatus};

/// A compact summary of a completed or in-flight run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RunSummary {
    /// The unique run identifier.
    pub run_id: RunId,

    /// The current run status.
    pub status: RunStatus,

    /// The selected model, if one has been resolved.
    pub model_id: Option<ModelId>,

    /// The timestamp at which the run started, if known.
    pub started_at: Option<Timestamp>,

    /// The timestamp at which the run ended, if known.
    pub completed_at: Option<Timestamp>,

    /// Token usage recorded for the run, if available.
    pub usage: Option<TokenUsage>,

    /// The ordered round summaries observed for the run.
    pub rounds: Vec<RoundSummary>,
}
