use std::fmt;

use serde::{Deserialize, Serialize};

/// A stable identifier for a single operation issued to the Nexo Gateway.
///
/// This is used to correlate operations with responses and events,
/// especially for asynchronous processing.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct OperationId(String);

impl OperationId {
    /// Creates a new operation identifier from an owned string.
    ///
    /// # Arguments
    ///
    /// * `value` - The unique operation identifier.
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the operation identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for OperationId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for OperationId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for OperationId {
    fn from(value: &str) -> Self {
        Self::new(value.to_owned())
    }
}
