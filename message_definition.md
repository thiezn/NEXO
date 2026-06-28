b# The Loop

in Nexo Node, I think. i want a join_set, this avoids needing a channel. It all does lead to a gap in my protocol/events/requests which i need to tackle if i want a good architecture. 

Decided on the following: 


# Messaging

The protocol distinguishes between **requests**, **responses**, and **events**.

A **request** asks the receiving component to perform an action or return information. Requests may represent either commands or queries. Every request expects exactly one response.

Enum values for requests should be named in the imperative form, e.g., `RunInference`, `Cancel`, `GetState`, etc.

A **response** communicates the immediate protocol outcome of a request. A response is always correlated to a prior request and is one of:


- **completed** — the request was processed immediately and the final result is included;
- **accepted** — the request was accepted for asynchronous processing;
- **failed** — the request could not be accepted or processed.

If a request is accepted for asynchronous processing, the response MUST include a `operation_id` that identifies the operation lifecycle for subsequent correlation.

The naming of a Response enum value **must** must be grouped under the corresponding Request name, e.g., `RunInference` request will have a `RunInference` response.

An **event** is an asynchronous notification emitted outside the initial request/response exchange. Events may be:

- **correlated**, meaning they belong to a previously accepted request and therefore include `operation_id`; or
- **unsolicited**, meaning they describe an independent state change such as lifecycle, status, capacity, or presence updates.

Correlated events are used for progress updates, intermediate outputs, lifecycle milestones, and final asynchronous completion, failure, or cancellation.

The naming of events related to a specific requests (Correlated events) **must** match the corresponding Request enum value, e.g., `RunInference` request will have `RunInferenceEvent` events. The payload of the event will be another enum that describes any specific event types related to the request, e.g., `Chunk`, `Completed`, `Failed`, etc.

The naming of Unsolicited events **must** be descriptive of the state change they represent, e.g., `MetricsEvent`

## pseudo-code

The top-level message definitions can be represented as follows:

```rust

/// This is the top level wrapper for all messages sent through the WebSocket connection.
pub struct Envelope<T> {
    pub message_id: MessageId,
    pub payload: T,
}

/// NexoResponse is a helper type for any kind of response.
/// 
/// Wrap specific directional responses in this type for consistent
/// response handling.
/// 
/// NOTE: We do not have a NexoRequest type because requests are always
/// specific to the action being requested.
#[derive(Debug)]
pub enum NexoResponse<T, E> {

    /// The request was processed immediately and the final result is included.
    Completed {
        /// The original operation_id
        operation_id: OperationId,
        result: T,
    },

    /// The request was accepted for asynchronous processing.
    Accepted {
        operation_id: OperationId,
    },

    /// The request could not be accepted or processed.
    Failed {
        operation_id: OperationId,
        error: E,
    },
}


/// NexoEvent is a helper type for any kind of event.
/// 
/// Wrap specific directional events in this type for consistent
/// event handling.
/// 
/// NOTE: We do not have a NexoRequest type because requests are always
/// specific to the action being requested.
#[derive(Debug)]
pub enum NexoEvent<T> {

    /// The event belongs to a previously accepted request and therefore includes `operation_id`.
    Correlated {
        operation_id: OperationId,
        event: T,
    },

    /// The event describes an independent state change such as lifecycle, status, capacity, or presence updates.
    Unsolicited {
        event: T,
    },
}


/// Example of a specific request payload to ask for running inference.
/// 
/// note that we will NOT need a RunInferenceResponse type as it will map to a NexoResponse::Accepted. Then
/// actual results will come in through events.
pub struct RunInferenceRequest {
    pub operation_id: OperationId,
    pub model: String,
    pub input: serde_json::Value,
}


/// Example of a specific request payload to ask to cancel a previously accepted inference request.
pub struct CancelInferenceRequest {
    pub operation_id: OperationId,
}

/// Example of a specific request payload to ask for creating a session.
/// 
/// These will be wrapped in a NexoEvent::Correlated event with a operation_id, and the actual results will come in through events.
enum InferenceEvent {    
    Chunk { 
        /// The sequence number of the chunk, starting from 0. This allows the client 
        /// to reassemble the chunks in the correct order.
        seq: usize,
        output: String     
    },
    Completed(
        total_chunks: usize
    ),
    Failed(String),
}


pub enum ClientToGatewayRequest {
    RunInference(RunInferenceRequest),

    // example
    SessionCreate(SessionCreateRequest),

    // etc...
}

pub enum GatewayToClientEvent {
    Inference(InferenceEvent),
    Session(SessionEvent),
    System(SystemEvent),
}



pub type GatewayResponse<T> = NexoResponse<T, GatewayError>;

pub enum GatewayToClientResponse {
    RunInference(GatewayResponse<RunInferenceResult>),
}


```
