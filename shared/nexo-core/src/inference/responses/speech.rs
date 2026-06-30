use serde::{Deserialize, Serialize};

use crate::{AudioFormat, MediaSource};

/// The response returned for a speech generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SpeechGenerationResponse {
    /// The generated speech audio payload.
    pub audio: GeneratedAudio,
}

/// A transport-safe generated audio payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GeneratedAudio {
    /// The generated audio content.
    pub source: MediaSource,

    /// The format used for the generated audio.
    pub format: AudioFormat,

    /// The audio sample rate, in hertz, if known.
    pub sample_rate_hz: Option<u32>,

    /// The number of audio channels, if known.
    pub channel_count: Option<u16>,
}
