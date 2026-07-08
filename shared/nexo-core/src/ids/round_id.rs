use serde::{Deserialize, Serialize};

/// A stable identifier for a single round within a run.
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
pub struct RoundId(usize);

impl RoundId {
    /// Creates a new round identifier.
    pub fn new() -> Self {
        Self(0)
    }
}
