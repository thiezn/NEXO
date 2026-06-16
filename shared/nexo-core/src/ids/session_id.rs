use std::fmt;

use serde::{Deserialize, Serialize};

/// A stable identifier for a long-lived conversation or inference session.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Creates a new session identifier from an owned string.
    ///
    /// # Arguments
    ///
    /// * `value` - The unique session identifier.
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the session identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self::new(value.to_owned())
    }
}
