use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A stable identifier for a single operation issued to the Nexo Gateway.
///
/// This is used to correlate operations with responses and events,
/// especially for asynchronous processing.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct OperationId(Uuid);

impl OperationId {
    /// Creates a new operation identifier with a time-sortable UUID v7.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Creates a new operation identifier from an owned string.
    pub fn from_string(value: String) -> Self {
        Self(Uuid::parse_str(&value).expect("Invalid UUID string"))
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0.to_string()
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<String> for OperationId {
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

impl From<&str> for OperationId {
    fn from(value: &str) -> Self {
        Self::from_string(value.to_owned())
    }
}
