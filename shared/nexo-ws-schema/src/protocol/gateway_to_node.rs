use super::CancelRequest;
use crate::NexoResponse;
use nexo_core::{InferenceRequest, ModelId, OperationId, ToolCall};
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

/// The messages that can be sent from a Gateway to a Node.
#[derive(Debug, IntoStaticStr, Serialize, Deserialize)]
pub enum GatewayToNodeMessage {
    /// A NexoResponse::Completed to a Node request to indicate the Connect was completed
    /// synchronously and the Node is now connected to the gateway.
    Connect(NexoResponse),

    /// A NexoResponse::Completed to a Node request to indicate the Disconnect was completed
    /// synchronously and the Node is now disconnected from the gateway.
    ///
    /// In the future this could change to a NexoResponse::Accepted to indicate the gateway is
    /// processing the disconnect gracefully and the Node should wait for a DisconnectCompleted event
    /// to be sent before the Node can consider itself fully disconnected.
    Disconnect(NexoResponse),

    /// A request to load a model on the Node with the specified parameters.
    LoadModel {
        /// The unique identifier for the operation to load the model.
        operation_id: OperationId,

        /// The unique identifier of the model to be loaded.
        model_id: ModelId,
    },

    /// A request to unload a model on the Node with the specified parameters.
    UnloadModel {
        /// The unique identifier for the operation to unload the model.
        operation_id: OperationId,

        /// The unique identifier of the model to be unloaded.
        model_id: ModelId,
    },

    /// A request to run an inference operation with the specified parameters.
    StartInferenceRun {
        /// The unique identifier for the inference operation to be started.
        operation_id: OperationId,

        /// The inference request parameters for the operation.
        request: InferenceRequest,
    },

    /// A request to cancel a previously submitted operation.
    Cancel(CancelRequest),

    /// A request to execute a tool on the Node with the specified parameters.
    ///
    /// Note that, although a Node actually runs inference operations which determine
    /// if a tool needs to be called, these inference tool call results are always routed
    /// back to the gateway, which makes the final decision on where to execute the tool.
    ExecuteTool {
        /// The unique identifier for the operation to execute the tool.
        operation_id: OperationId,

        /// The tool call request parameters for the operation.
        tool_call: ToolCall,
    },
}
