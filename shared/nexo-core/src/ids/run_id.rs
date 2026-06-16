use std::fmt;

use serde::{Deserialize, Serialize};

/// A stable identifier for a gateway-level run lifecycle.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    /// Creates a new run identifier from an owned string.
    ///
    /// # Arguments
    ///
    /// * `value` - The unique run identifier.
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the run identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for RunId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for RunId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RunId {
    fn from(value: &str) -> Self {
        Self::new(value.to_owned())
    }
}
