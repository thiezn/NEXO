use nexo_core::OperationId;
use serde::{Deserialize, Serialize};

/// The command to cancel a previously submitted request.
#[derive(Debug, Serialize, Deserialize)]
pub struct CancelRequest {
    /// The operation_id of the request to cancel.
    pub operation_id: OperationId,
}
