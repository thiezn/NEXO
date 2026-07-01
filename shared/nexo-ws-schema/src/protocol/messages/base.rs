use crate::{Error, Result};
use nexo_core::OperationId;
use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;
use tracing::{error, info};

/// NexoResponse is a helper type for any kind of response.
///
/// Wrap specific directional responses in this type for consistent
/// response handling.
///
/// NOTE: We do not have a NexoRequest type because operations are always
/// specific to the action being performed.
#[derive(Debug, IntoStaticStr, Serialize, Deserialize, PartialEq, Eq)]
pub enum NexoResponse<T = (), E = Error> {
    /// The operation was processed immediately and the final result is included.
    Completed {
        /// The original operation_id
        operation_id: OperationId,

        /// The result of processing the operation, if successful.
        ///
        /// Often we don't need any result data for a completed operation,
        /// so the default type is `()`.
        result: T,
    },

    /// The operation was accepted for asynchronous processing.
    Accepted {
        /// The original operation_id
        operation_id: OperationId,
    },

    /// The operation could not be accepted or processed.
    Failed {
        /// The original operation_id
        operation_id: OperationId,
        /// The error that occurred
        error: E,
    },
}

impl NexoResponse<(), Error> {
    /// Helper function to parse the Response result and log the outcome.
    ///
    /// This is useful for handling responses in a consistent manner,
    /// especially when the result type is not needed.
    pub fn result(&self) -> Result {
        let message_type: &'static str = self.into();
        match self {
            NexoResponse::Completed { operation_id, .. } => {
                info!(operation_id = %operation_id, message_type = message_type, "request completed");
                Ok(())
            }
            NexoResponse::Accepted { operation_id, .. } => {
                info!(
                    operation_id = %operation_id,
                    message_type = message_type,
                    "request accepted for asynchronous processing"
                );
                Ok(())
            }
            NexoResponse::Failed {
                operation_id,
                error,
                ..
            } => {
                error!(operation_id = %operation_id, message_type = message_type, "request failed: {error}");
                // Err(error.clone())
                Err(Error::ResponseFailed {
                    operation_id: operation_id.clone(),
                    error: error.to_string(),
                })
            }
        }
    }

    /// Helper function to create a Completed response without any result data.
    pub fn completed(operation_id: OperationId) -> Self {
        NexoResponse::Completed {
            operation_id,
            result: (),
        }
    }

    /// Helper function to create an Accepted response.
    pub fn accepted(operation_id: OperationId) -> Self {
        NexoResponse::Accepted { operation_id }
    }

    /// Helper function to check if the response is a Completed response.
    ///
    /// Useful for cases when we want to ensure the remote side completed the response
    /// synchronously and not accepted it for asynchronous processing.
    pub fn is_completed(&self) -> Result {
        match self {
            NexoResponse::Completed { .. } => Ok(()),
            NexoResponse::Accepted { operation_id } => Err(Error::ExpectedCompletedResponse {
                operation_id: operation_id.clone(),
                error: "request accepted for asynchronous processing".into(),
            }),
            NexoResponse::Failed {
                operation_id,
                error,
                ..
            } => Err(Error::ExpectedCompletedResponse {
                operation_id: operation_id.clone(),
                error: format!("request failed: {error}"),
            }),
        }
    }

    /// Helper function to check if the response is an Accepted response.
    ///
    /// Useful for cases when we want to ensure the remote side accepted the response for asynchronous processing
    /// and not completed it synchronously.
    pub fn is_accepted(&self) -> Result {
        match self {
            NexoResponse::Completed { operation_id, .. } => Err(Error::ExpectedAcceptedResponse {
                operation_id: operation_id.clone(),
                error: "request completed immediately".into(),
            }),
            NexoResponse::Accepted { .. } => Ok(()),
            NexoResponse::Failed {
                operation_id,
                error,
                ..
            } => Err(Error::ExpectedAcceptedResponse {
                operation_id: operation_id.clone(),
                error: format!("request failed: {error}"),
            }),
        }
    }
}

/// NexoEvent is a helper type for any kind of event.
///
/// Wrap specific directional events in this type for consistent
/// event handling.
///
/// NOTE: We do not have a NexoRequest type because operations are always
/// specific to the action being operationed.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NexoEvent<T> {
    /// The event belongs to a previously accepted operation and therefore includes `operation_id`.
    Correlated {
        /// The original operation_id
        operation_id: OperationId,
        /// The event data
        event: T,
    },

    /// The event describes an independent state change such as lifecycle, status, capacity, or presence updates.
    Unsolicited {
        /// The event data
        event: T,
    },
}

impl NexoEvent<()> {
    /// Helper function to create a Correlated event without any event data.
    pub fn correlated(operation_id: OperationId) -> Self {
        NexoEvent::Correlated {
            operation_id,
            event: (),
        }
    }

    /// Helper function to create an Unsolicited event.
    pub fn unsolicited() -> Self {
        NexoEvent::Unsolicited { event: () }
    }
}
