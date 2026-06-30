use serde::{Deserialize, Serialize};

/// Monotonic sequence number for streamed inference updates.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
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
pub struct StreamSeq(u64);

impl StreamSeq {
    /// Returns the first sequence number in a stream.
    pub const fn first() -> Self {
        Self(0)
    }

    /// Returns the next sequence number.
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Returns the raw sequence number.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for StreamSeq {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<StreamSeq> for u64 {
    fn from(value: StreamSeq) -> Self {
        value.0
    }
}

/// Stable index for one generated artifact within an inference output.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
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
pub struct ArtifactIndex(u32);

impl ArtifactIndex {
    /// Creates an artifact index from its raw value.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw artifact index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for ArtifactIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<ArtifactIndex> for u32 {
    fn from(value: ArtifactIndex) -> Self {
        value.0
    }
}

/// Absolute byte offset for streamed textual output.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
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
pub struct OutputOffsetBytes(u64);

impl OutputOffsetBytes {
    /// Creates an output offset from its raw byte position.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw byte offset.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for OutputOffsetBytes {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<OutputOffsetBytes> for u64 {
    fn from(value: OutputOffsetBytes) -> Self {
        value.0
    }
}
