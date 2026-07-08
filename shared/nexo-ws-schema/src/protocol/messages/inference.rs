use nexo_core::{InferenceMeta, InferenceOutputDelta, InferenceUpdate, ModelId, StreamSeq};
use serde::{Deserialize, Serialize};

/// The events that can be emitted related to an operation started by a StartInferenceRun request.
///
/// The originating nexo node will emit these events to the gateway. The gateway
/// will handle them (e.g. update internal state, etc) and forward them to the Client.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum InferenceRunEvent {
    /// The inference request has been accepted and actual processing has begun on a nexo-node.
    RunStarted {
        /// The resolved execution identity.
        meta: InferenceMeta,
    },

    /// A single round of inference has completed for the inference request. This event is emitted
    /// for each round of inference that completes, and can be used to track the progress of
    /// the inference request.
    RoundCompleted {
        /// The resolved execution identity.
        meta: InferenceMeta,
    },

    /// Progressive output has been generated for the inference request.
    Output {
        /// The resolved execution identity.
        meta: InferenceMeta,

        /// The sequence number for this output event.
        seq: StreamSeq,

        /// The typed output generated for the inference request.
        output: InferenceOutputDelta,
    },

    /// The inference request has completed successfully and all output has been generated.
    RunCompleted {
        /// The resolved execution identity.
        meta: InferenceMeta,

        /// The total number of output events that were generated for the inference request.
        total_outputs: StreamSeq,
    },

    /// The inference request has been cancelled before completion.
    Cancelled {
        /// The resolved execution identity.
        meta: InferenceMeta,

        /// Optional human-readable cancellation reason.
        reason: Option<String>,
    },

    /// The inference request has failed with an error.
    Failed {
        /// The resolved execution identity.
        meta: InferenceMeta,

        /// The error message describing the failure.
        error: String,
    },
}

impl From<InferenceUpdate> for InferenceRunEvent {
    fn from(update: InferenceUpdate) -> Self {
        match update {
            InferenceUpdate::Started(update) => Self::RunStarted { meta: update.meta },
            InferenceUpdate::Progress(update) => Self::Output {
                meta: update.meta,
                seq: update.seq,
                output: update.output,
            },
            InferenceUpdate::Completed(update) => Self::RunCompleted {
                meta: update.meta,
                total_outputs: StreamSeq::first(),
            },
            InferenceUpdate::Cancelled(update) => Self::Cancelled {
                meta: update.meta,
                reason: update.reason,
            },
            InferenceUpdate::Failed(update) => Self::Failed {
                meta: update.meta,
                error: update.error,
            },
        }
    }
}

/// The events emitted related to an operation started by a LoadModel request.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LoadModelEvent {
    /// The load model request has been accepted and actual processing has begun on a nexo-node.
    Started {
        /// The model ID that is in the process of being loaded.
        model_id: ModelId,
    },

    /// The load model request has completed successfully and the model is now available for inference.
    Completed {
        /// The model ID that has been successfully loaded.
        model_id: ModelId,
    },

    /// The load model request has failed with an error.
    Failed {
        /// The model_id that was attempted to be loaded.
        model_id: ModelId,

        /// The error message describing the failure.
        error: String,
    },
}

/// The events emitted related to an operation started by a UnloadModel request.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum UnloadModelEvent {
    /// The unload model request has been accepted and actual processing has begun on a nexo-node.
    Started {
        /// The model ID that is in the process of being unloaded.
        model_id: ModelId,
    },

    /// The unload model request has completed successfully and the model is now unloaded.
    Completed {
        /// The model ID that has been successfully unloaded.
        model_id: ModelId,
    },

    /// The unload model request has failed with an error.
    Failed {
        /// The model_id that was attempted to be unloaded.
        model_id: ModelId,

        /// The error message describing the failure.
        error: String,
    },
}
