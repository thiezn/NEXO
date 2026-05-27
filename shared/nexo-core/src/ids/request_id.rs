use std::fmt;

use serde::{Deserialize, Serialize};

/// A stable identifier for a single inference request.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct RequestId(String);

impl RequestId {
    /// Creates a new request identifier from an owned string.
    ///
    /// # Arguments
    ///
    /// * `value` - The unique request identifier.
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the request identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for RequestId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for RequestId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self::new(value.to_owned())
    }
}
