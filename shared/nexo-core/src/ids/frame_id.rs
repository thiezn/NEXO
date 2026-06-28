use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A stable identifier for a WebSocket frame.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
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

    // /// Returns the frame identifier as a string slice.
    // pub fn as_str(&self) -> &str {
    //     self.0.to_string().as_str()
    // }

    /// Consumes the identifier and returns the owned string value.
    pub fn into_string(self) -> String {
        self.0.to_string()
    }
}

// impl AsRef<str> for FrameId {
//     fn as_ref(&self) -> &str {
//         self.as_str()
//     }
// }

// impl fmt::Display for FrameId {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.write_str(self.as_str())
//     }
// }

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
