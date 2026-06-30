use serde::{Deserialize, Serialize};

use crate::message::ConversationMessage;

use super::super::finish::FinishReason;
use super::super::usage::{PerformanceMetrics, TokenUsage};

/// The final response for a completed multimodal generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MultiModalResponse {
    /// The final assistant message.
    pub message: ConversationMessage,

    /// The full reasoning content captured during generation, if requested.
    pub reasoning: Option<String>,

    /// The final reason generation ended.
    pub finish_reason: FinishReason,

    /// Token usage recorded for the completed response.
    pub usage: Option<TokenUsage>,

    /// Optional performance metrics recorded for the response.
    pub performance: Option<PerformanceMetrics>,
}
