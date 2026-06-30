use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::message::MessageRole;
use crate::tools::ToolCallDelta;

use super::update::InferenceUpdate;

/// A boxed update stream returned by an inference engine.
pub type InferenceStream = BoxStream<'static, Result<InferenceUpdate>>;

/// A streamed delta for an in-progress generation request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateDelta {
    /// The assistant role emitted with the delta, if this chunk establishes one.
    pub role: Option<MessageRole>,

    /// The incremental user-visible content produced by the model.
    pub content_delta: Option<String>,

    /// The incremental reasoning content produced by the model, if requested.
    pub reasoning_delta: Option<String>,

    /// The streamed tool call updates produced by the model.
    pub tool_call_deltas: Vec<ToolCallDelta>,
}
