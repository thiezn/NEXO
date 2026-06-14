use crate::MediaSource;
use crate::ModelSelection;
use serde::{Deserialize, Serialize};

/// The requested spoken language for speech synthesis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SpeechLanguage {
    /// English speech output.
    #[default]
    English,

    /// Dutch speech output.
    Dutch,
}

/// The requested audio encoding for synthesized speech output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// Raw pulse-code modulation samples.
    Pcm,

    /// WAV-encoded audio.
    Wav,

    /// MP3-encoded audio.
    Mp3,
}

/// A request to synthesize speech from text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SpeechGenerationPayload {
    /// The text to synthesize into speech.
    pub text: String,

    /// The requested spoken language.
    #[serde(default)]
    pub language: SpeechLanguage,

    /// The requested voice label, if the runtime supports voice selection.
    pub voice: Option<String>,

    /// The desired audio format for the output.
    pub format: AudioFormat,

    /// The desired output sample rate, in hertz, if supported.
    pub sample_rate_hz: Option<u32>,

    /// The requested speaking speed multiplier, if supported.
    pub speed: Option<f32>,
}

/// A transport-safe generated audio payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
