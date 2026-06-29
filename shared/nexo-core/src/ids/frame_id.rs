use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A stable identifier for a WebSocket frame.
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
pub struct FrameId(Uuid);

impl FrameId {
    /// Creates a new frame identifier with a time-sortable UUID v7.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Creates a new frame identifier from an owned string.
    pub fn from_string(value: String) -> Self {
        Self(Uuid::parse_str(&value).expect("Invalid UUID string"))
    }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0.to_string()
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<String> for FrameId {
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

impl From<&str> for FrameId {
    fn from(value: &str) -> Self {
        Self::from_string(value.to_owned())
    }
}
