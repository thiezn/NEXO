use crate::error::Result;
use crate::inference::{InferenceRequest, InferenceStream};

/// A service capable of executing shared `nexo-core` inference requests.
pub trait InferenceEngine: Send + Sync {
    /// Submits a new inference request and returns a stream of responses.
    ///
    /// # Arguments
    ///
    /// * `request` - The inference request to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the request cannot be accepted for execution.
    fn submit(&self, request: InferenceRequest) -> Result<InferenceStream>;
}
