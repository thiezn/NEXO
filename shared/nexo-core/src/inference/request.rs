use super::{InferenceIntent, InferenceOperation, InferenceOperationKind};
use crate::{ConversationMessage, ModelId, OperationId, RoundId, RunId, SessionId};
use serde::{Deserialize, Serialize};

/// A unified representation of a fully prepared request that can be executed by a node.
///
/// - The type of inference request can be any modality
/// - It fully conveys the intent of the user, and includes any system prompt and
///   model selection logic that has been applied to the request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct InferenceRequest {
    /// The unique identifier for the inference request.
    pub operation_id: OperationId,

    /// The session identifier for the request, used to correlate related requests.
    pub session_id: SessionId,

    /// The run identifier used by higher-level orchestration to correlate requests
    /// across a workflow.
    pub run_id: RunId,

    /// The round identifier used to correlate a single loop iteration.
    pub round_id: RoundId,

    /// The model identifier for the request.
    pub model_id: ModelId,

    /// The specific inference operation being requested.
    #[serde(flatten)]
    pub operation: InferenceOperation,
}

impl InferenceRequest {
    /// Returns the kind of inference operation carried by this request.
    pub fn operation_kind(&self) -> InferenceOperationKind {
        self.operation.kind()
    }

    /// Creates a new InferenceRequest from an InferenceIntent and a ModelId.
    ///
    /// # Arguments
    ///
    /// * `intent` - The InferenceIntent to convert into an InferenceRequest.
    /// * `model_id` - The ModelId to use for the InferenceRequest.
    /// * `conversation_prefix` - A vector of ConversationMessages to prepend to
    ///    the conversation in the InferenceRequest.
    pub fn from_intent(
        intent: &InferenceIntent,
        model_id: ModelId,
        mut conversation_prefix: Vec<ConversationMessage>,
    ) -> Self {
        let mut operation = intent.operation.clone();
        if let InferenceOperation::MultiModal(ref mut payload) = operation {
            conversation_prefix.append(&mut payload.conversation.messages);
            payload.conversation.messages = conversation_prefix;
        }

        Self {
            operation_id: intent.operation_id,
            session_id: intent.session_id,
            run_id: RunId::new(),
            round_id: RoundId::new(),
            model_id,
            operation,
        }
    }
}
