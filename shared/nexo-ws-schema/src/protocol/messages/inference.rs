use crate::Error;
use nexo_core::{ModelId, OperationId, RoundId, RunId};
use serde::{Deserialize, Serialize};

/// The events that can be emitted related to an operation started by a StartInferenceRun request.
///
/// The originating nexo node will emit these events to the gateway. The gateway
/// will handle them (e.g. update internal state, etc) and forward them to the Client.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InferenceEvent {
    /// The inference request has been accepted and actual processing has begun on a nexo-node.
    RunStarted {
        /// The operation_id of the inference request that has started.
        operation_id: OperationId,

        /// The ID of the inference run that has started.
        run_id: RunId,
    },

    /// A single round of inference has completed for the inference request. This event is emitted
    /// for each round of inference that completes, and can be used to track the progress of
    /// the inference request.
    RoundCompleted {
        /// The operation_id of the inference request that has started.
        operation_id: OperationId,

        /// The ID of this specific run
        run_id: RunId,

        /// The inference round id
        round_id: RoundId,
    },

    /// A chunk of output has been generated for the inference request.
    ///
    /// TODO: Do we need to differentiate between thinking and normal output?
    /// I assume that both thinking and normal output can stream,
    /// perhaps we need ThinkingChunk, NormalChunk, etc? Perhaps we need to
    /// rename it as Chunk might not be descriptive enough?
    ///
    /// What about inferencerequests for images/audio/video, they will also
    /// stream and return content. Can we capture this all in a Chunk enum
    /// or do we need separate ones?
    Chunk {
        /// The sequence number of the chunk, starting from 0. This allows the client
        /// to reassemble the chunks in the correct order.
        seq: usize,

        /// The output generated for the inference request.
        output: String,
    },

    /// The inference request has completed successfully and all output has been generated.
    RunCompleted {
        /// The total number of chunks that were generated for the inference request.
        total_chunks: usize,
    },

    /// The inference request has been cancelled before completion.
    Cancelled,

    /// The inference request has failed with an error.
    Failed {
        /// The error message describing the failure.
        error: String,
    },
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
