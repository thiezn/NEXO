use serde::{Deserialize, Serialize};

use crate::message::MediaSource;
use crate::tools::ToolCallDelta;
use crate::{AudioFormat, FinishReason, MessageRole, ModelId, OperationId, RoundId, RunId};

use super::InferenceOperationKind;
use super::meta::InferenceMeta;
use super::ordering::{ArtifactIndex, OutputOffsetBytes, StreamSeq};
use super::output::InferenceOutput;
use super::responses::EmbeddingVector;
use super::responses::GeneratedImage;
use super::usage::TokenUsage;

/// A streamed update emitted while an inference operation executes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceUpdate {
    /// The runtime has started executing the request.
    Started(InferenceStarted),

    /// The runtime emitted progressive output.
    Progress(InferenceProgress),

    /// The runtime completed successfully.
    Completed(InferenceCompleted),

    /// The runtime was cancelled before completion.
    Cancelled(InferenceCancelled),

    /// The runtime failed before producing a successful final output.
    Failed(InferenceFailed),
}

impl InferenceUpdate {
    /// Creates a started update.
    pub fn started(meta: InferenceMeta) -> Self {
        Self::Started(InferenceStarted { meta })
    }

    /// Creates a progress update.
    pub fn progress(meta: InferenceMeta, seq: StreamSeq, output: InferenceOutputDelta) -> Self {
        Self::Progress(InferenceProgress { meta, seq, output })
    }

    /// Creates a completed update.
    pub fn completed(meta: InferenceMeta, final_output: InferenceOutput) -> Self {
        Self::Completed(InferenceCompleted { meta, final_output })
    }

    /// Creates a failed update.
    pub fn failed(meta: InferenceMeta, error: String) -> Self {
        Self::Failed(InferenceFailed { meta, error })
    }

    /// Creates a cancelled update.
    pub fn cancelled(meta: InferenceMeta, reason: Option<String>) -> Self {
        Self::Cancelled(InferenceCancelled { meta, reason })
    }

    /// Returns metadata for updates that have resolved execution identity.
    pub const fn meta(&self) -> &InferenceMeta {
        match self {
            Self::Started(update) => &update.meta,
            Self::Progress(update) => &update.meta,
            Self::Completed(update) => &update.meta,
            Self::Cancelled(update) => &update.meta,
            Self::Failed(update) => &update.meta,
        }
    }

    /// Returns the operation identifier.
    pub fn operation_id(&self) -> OperationId {
        self.meta().operation_id
    }

    /// Returns the run identifier.
    pub fn run_id(&self) -> RunId {
        self.meta().run_id
    }

    /// Returns the round identifier.
    pub fn round_id(&self) -> RoundId {
        self.meta().round_id.clone()
    }

    /// Returns the resolved model identifier.
    pub fn model_id(&self) -> &ModelId {
        &self.meta().model_id
    }

    /// Returns true when this update terminates the operation stream.
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed(_) | Self::Cancelled(_) | Self::Failed(_)
        )
    }

    /// Returns true when this update carries progressive output.
    pub const fn is_progress(&self) -> bool {
        matches!(self, Self::Progress(_))
    }

    /// Returns the operation kind implied by the update payload, if known.
    pub fn operation_kind_hint(&self) -> Option<InferenceOperationKind> {
        match self {
            Self::Progress(update) => Some(update.output.operation_kind()),
            Self::Completed(update) => Some(update.final_output.operation_kind()),
            Self::Started(_) | Self::Cancelled(_) | Self::Failed(_) => None,
        }
    }

    /// Returns true when this update is known to belong to the provided operation kind.
    pub fn matches_operation_kind(&self, kind: InferenceOperationKind) -> bool {
        self.operation_kind_hint()
            .is_none_or(|operation_kind| operation_kind == kind)
    }
}

/// Metadata emitted when inference execution starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceStarted {
    /// The resolved execution identity.
    pub meta: InferenceMeta,
}

/// Progressive output from an inference operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceProgress {
    /// The resolved execution identity.
    pub meta: InferenceMeta,

    /// The sequence number for this progressive update.
    pub seq: StreamSeq,

    /// The progressive output payload.
    pub output: InferenceOutputDelta,
}

/// Successful terminal update from an inference operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceCompleted {
    /// The resolved execution identity.
    pub meta: InferenceMeta,

    /// The successful final output.
    pub final_output: InferenceOutput,
}

/// Cancelled terminal update from an inference operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceCancelled {
    /// The resolved execution identity.
    pub meta: InferenceMeta,

    /// Optional human-readable cancellation reason.
    pub reason: Option<String>,
}

/// Failed terminal update from an inference operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InferenceFailed {
    /// The resolved execution identity.
    pub meta: InferenceMeta,

    /// Human-readable failure message.
    pub error: String,
}

/// Progressive output emitted by an inference operation.
///
/// This will eventually be combined into a InferenceOutput
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum InferenceOutputDelta {
    /// Conversational or multimodal generation deltas.
    MultiModal(MultiModalDelta),

    /// Partial or complete embedding output.
    Embedding(EmbeddingDelta),

    /// Progressive generated image output.
    Image(ImageDelta),

    /// Progressive generated speech output.
    Speech(SpeechDelta),

    /// Partial or complete tokenized output.
    Tokenization(TokenizationDelta),

    /// Partial detokenized text output.
    Detokenization(DetokenizationDelta),
}

impl InferenceOutputDelta {
    /// Returns the operation kind that produced this progressive output.
    pub const fn operation_kind(&self) -> InferenceOperationKind {
        match self {
            Self::MultiModal(_) => InferenceOperationKind::MultiModal,
            Self::Embedding(_) => InferenceOperationKind::Embed,
            Self::Image(_) => InferenceOperationKind::GenerateImage,
            Self::Speech(_) => InferenceOperationKind::GenerateSpeech,
            Self::Tokenization(_) => InferenceOperationKind::Tokenize,
            Self::Detokenization(_) => InferenceOperationKind::Detokenize,
        }
    }
}

/// A streamed delta for an in-progress multimodal generation request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MultiModalDelta {
    /// The assistant role emitted with the delta, if this chunk establishes one.
    pub role: Option<MessageRole>,

    /// The incremental user-visible content produced by the model.
    pub content_delta: Option<String>,

    /// The incremental reasoning content produced by the model, if requested.
    pub reasoning_delta: Option<String>,

    /// The streamed tool call updates produced by the model.
    pub tool_call_deltas: Vec<ToolCallDelta>,

    /// Token usage recorded with this delta, if available.
    pub usage: Option<TokenUsage>,

    /// The finish reason, if this delta terminates multimodal generation.
    pub finish_reason: Option<FinishReason>,
}

/// Partial or complete embedding output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingDelta {
    /// The embedding vectors emitted by this update.
    pub vectors: Vec<EmbeddingVector>,
}

/// Progressive generated image output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageDelta {
    /// The zero-based generated artifact index.
    pub index: ArtifactIndex,

    /// The sequence number within this image artifact.
    pub seq: StreamSeq,

    /// The generated image content or reference.
    pub source: MediaSource,

    /// The optional media type of the generated image.
    pub media_type: Option<String>,

    /// The generated image width, in pixels, if known.
    pub width: Option<u32>,

    /// The generated image height, in pixels, if known.
    pub height: Option<u32>,

    /// Whether this segment completes the image artifact.
    pub is_final_segment: bool,
}

impl From<GeneratedImage> for ImageDelta {
    fn from(value: GeneratedImage) -> Self {
        Self {
            index: ArtifactIndex::new(value.index as u32),
            seq: StreamSeq::first(),
            source: value.source,
            media_type: value.media_type,
            width: value.width,
            height: value.height,
            is_final_segment: true,
        }
    }
}

/// Progressive generated speech output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SpeechDelta {
    /// The sequence number within the speech stream.
    pub seq: StreamSeq,

    /// The generated audio content or reference.
    pub source: MediaSource,

    /// The audio format used by the generated speech payload.
    pub format: AudioFormat,

    /// The audio sample rate, in hertz, if known.
    pub sample_rate_hz: Option<u32>,

    /// The number of audio channels, if known.
    pub channel_count: Option<u16>,

    /// Whether this segment completes the speech artifact.
    pub is_final_segment: bool,
}

/// Partial or complete tokenized output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenizationDelta {
    /// The token ids emitted by this update.
    pub tokens: Vec<u32>,

    /// Optional byte offset in the original text input.
    pub offset_bytes: Option<OutputOffsetBytes>,
}

/// Partial detokenized text output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DetokenizationDelta {
    /// The detokenized text emitted by this update.
    pub text_delta: String,

    /// Optional byte offset in the assembled output text.
    pub offset_bytes: Option<OutputOffsetBytes>,
}
