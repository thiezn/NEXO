use super::{
    ExecuteToolEvent, InferenceRunEvent, LoadModelEvent, NexoEvent, NexoResponse, UnloadModelEvent,
};
use nexo_core::{NexoNodeMetrics, NodeProperties, ToolDefinition, ToolResult};
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// The messages that can be sent from a node to a gateway.
#[derive(Debug, IntoStaticStr, Serialize, Deserialize, PartialEq)]
pub enum NodeToGatewayMessage {
    /// Connect to the gateway and establish a session.
    ///
    /// TODO: ConnectParams should split into generic connect details and client specific params.
    /// Same for Node connect
    Connect(NodeProperties),

    /// Disconnect from the gateway and close the session gracefully.
    Disconnect,

    /// Current metrics of the Node, including cpu/gpu and memory usage.
    ///
    /// This includes metrics like cpu/gpu and memory usage.
    GetMetricsEvent(NexoEvent<NexoNodeMetrics>),

    /// A asynchronous NexoResponse::Accepted to a request to load a model on the Node with the
    /// specified parameters.
    LoadModel(NexoResponse),

    /// An event emitted for a load model request that was accepted for asynchronous processing.
    LoadModelEvent(NexoEvent<LoadModelEvent>),

    /// A asynchronous NexoResponse::Accepted to a request to unload a model on the Node with the
    /// specified parameters.
    UnloadModel(NexoResponse),

    /// An event emitted for a unload model request that was accepted for asynchronous processing.
    UnloadModelEvent(NexoEvent<UnloadModelEvent>),

    /// A asynchronous NexoResponse::Accepted to a request to start an inference run on the Node with the
    /// specified parameters.
    StartInferenceRun(NexoResponse),

    /// An event emitted for an inference request that was accepted for asynchronous processing.
    StartInferenceRunEvent(NexoEvent<InferenceRunEvent>),

    /// Register tools with the gateway, so that they can be used by other nodes and clients.
    RegisterTools(Vec<ToolDefinition>),

    /// Response to an ExecuteTool request, can be either a synchronous (Completed) or asynchronous (accepted)
    /// response depending on the tool and the parameters.
    ExecuteTool(NexoResponse<ToolResult>),

    /// An event emitted for an execute tool request that was accepted for asynchronous processing.
    ExecuteToolEvent(NexoEvent<ExecuteToolEvent>),

    /// A response to a CancelRequest, indicating whether the cancellation was successful or not.
    ///
    /// TODO: Having a generic cancel event is simple from a protocol perspective, but it will require
    /// the receiving node to maintain the mapping of an operation_id to the specific request
    /// type that was cancelled, and have to have storage of all required information to be able to cancel
    /// the request. It might be better to explicitly codify the Cancel requests per request type.
    Cancel(NexoResponse),

    /// An unknown failure occured during processing of a frame.
    UnknownFailure(NexoResponse),
}
