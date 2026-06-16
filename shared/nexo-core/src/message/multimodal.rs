use serde::{Deserialize, Serialize};

/// A transport-safe reference to binary media content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum MediaSource {
    /// Raw in-memory bytes.
    Bytes(Vec<u8>),

    /// Base64-encoded media content.
    Base64(String),

    /// A remote or local URL that points to the media.
    Url(String),
}

/// An image input attached to a conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageInput {
    /// The media source that contains the image bytes.
    pub source: MediaSource,

    /// The optional media type, such as `image/png`.
    pub media_type: Option<String>,
}

/// A video input attached to a conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct VideoInput {
    /// The media source that contains the video bytes.
    pub source: MediaSource,

    /// The optional media type, such as `video/mp4`.
    pub media_type: Option<String>,

    /// Optional frame timestamps, in milliseconds, when a sampled subset is used.
    pub frame_timestamps_ms: Vec<u64>,
}

/// An audio input attached to a conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AudioInput {
    /// The media source that contains the audio bytes.
    pub source: MediaSource,

    /// The optional media type, such as `audio/wav`.
    pub media_type: Option<String>,

    /// The sample rate, in hertz, if known.
    pub sample_rate_hz: Option<u32>,

    /// The number of audio channels, if known.
    pub channel_count: Option<u16>,
}
