use nexo_core::{
    CompactionRequest, InferenceIntent, InferenceRequest, InferenceRunEvent, LoadModelEvent,
    ModelId, NexoState, Node, OperationId, PeerId, UnloadModelEvent, User,
};
use strum::IntoStaticStr;

/// A message sent from NexoGateway to the NexoAgent.
#[derive(Debug, IntoStaticStr, PartialEq)]
pub enum NexoAgentInput {
    /// A new user has connected.
    UserConnected(User),
    /// A new node has connected.
    NodeConnected(Node),
    /// A user has disconnected.
    UserDisconnected(PeerId),
    /// A node has disconnected.
    NodeDisconnected(PeerId),
    /// A request to start a new inference run operation with the specified parameters.
    UserStartInferenceRun {
        /// The requesting user peer that owns the operation.
        requester: PeerId,
        /// The requested inference intent.
        intent: InferenceIntent,
    },
    /// A request to append additional instructions to an ongoing inference run operation.
    UserAppendInferenceInstructions {
        /// The unique identifier for the inference operation to which the instructions should be appended.
        operation_id: OperationId,
        /// The additional instructions to be appended to the ongoing inference operation.
        instructions: InferenceIntent,
    },
    /// A request to compact a given session.
    UserCompact(CompactionRequest),
    /// A model-load lifecycle event emitted by an authenticated node.
    NodeLoadModelEvent {
        /// Operation associated with the model load.
        operation_id: OperationId,
        /// Authenticated node peer that emitted the event.
        source_node_id: PeerId,
        /// Shared model-load event payload.
        event: LoadModelEvent,
    },
    /// A model-unload lifecycle event emitted by an authenticated node.
    NodeUnloadModelEvent {
        /// Operation associated with the model unload.
        operation_id: OperationId,
        /// Authenticated node peer that emitted the event.
        source_node_id: PeerId,
        /// Shared model-unload event payload.
        event: UnloadModelEvent,
    },
    /// A correlated inference event emitted by an authenticated node.
    NodeInferenceRunEvent {
        /// Authenticated node peer that emitted the event.
        source_node_id: PeerId,
        /// Operation associated with the event.
        operation_id: OperationId,
        /// Shared inference lifecycle event payload.
        event: InferenceRunEvent,
    },
    /// Retrieve the current state of the whole Nexo system.
    GetState {
        /// The requesting user peer that should receive the response.
        requester: PeerId,
        /// The operation identifier to preserve in the gateway response.
        operation_id: OperationId,
    },
}

/// An event sent from the Nexo Agent.
pub enum NexoAgentOutput {
    /// Ask a node to load a model for an inference operation.
    LoadModel {
        /// Target node for the command.
        node_peer_id: PeerId,
        /// Operation that requires the model.
        operation_id: OperationId,
        /// Model to load.
        model_id: ModelId,
    },
    /// Ask a node to unload a model before loading the selected target model.
    UnloadModel {
        /// Target node for the command.
        node_peer_id: PeerId,
        /// Operation waiting for memory to become available.
        operation_id: OperationId,
        /// Model to unload.
        model_id: ModelId,
    },
    /// Ask a node to begin a fully prepared inference request.
    StartInference {
        /// Target node for the command.
        node_peer_id: PeerId,
        /// Operation to execute.
        operation_id: OperationId,
        /// Fully prepared request with a concrete model selection.
        request: InferenceRequest,
    },
    /// Return the current state of the Nexo system.
    GetState {
        /// The requesting user peer that should receive the response.
        requester: PeerId,
        /// The operation identifier to preserve in the gateway response.
        operation_id: OperationId,
        /// Snapshot of the current in-memory system state.
        state: NexoState,
    },
}
