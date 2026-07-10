use nexo_core::{CompactionRequest, InferenceIntent, InferenceRequest, ModelId, NexoState, Node, OperationId, PeerId, User};
use nexo_ws_schema::{InferenceRunEvent, NexoEvent};
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
    /// An event indicating that a node loaded a model for an operation.
    ModelLoaded {
        /// The operation ID associated with the model load event.
        operation_id: OperationId,
        /// The node that has loaded the model.
        node: Node,
        /// The model ID that has been loaded.
        model_id: ModelId,
    },
    /// An event indicating that a node unloaded a model for an operation.
    ModelUnloaded {
        /// The operation ID associated with the model unload event.
        operation_id: OperationId,
        /// The node that has unloaded the model.
        node: Node,
        /// The model ID that has been unloaded.
        model_id: ModelId,
    },
    /// An event emitted from the node related to an inference run operation.
    NodeInferenceRunEvent(NexoEvent<InferenceRunEvent>),
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
    /// Send a fully prepared request to the node for processing.
    StartInferenceRun(Node, InferenceRequest),
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
