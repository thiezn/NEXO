use crate::ModelFamily;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A stable identifier for an inference model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[allow(missing_docs)]
pub enum ModelId {
    Gemma4E2bItQ5,
    Gemma412bIt,
    Gemma426bA4bItQ4,
    Gemma4E2bItUqffAfq2,
    Gemma4E2bItUqffAfq3,
    Gemma4E2bItUqffAfq4,
    Gemma4E2bItUqffAfq6,
    Gemma4E2bItUqffAfq8,
    Gemma4E2bItUqffQ2k,
    Gemma4E2bItUqffQ3k,
    Gemma4E2bItUqffQ4k,
    Gemma4E2bItUqffQ5k,
    Gemma4E2bItUqffQ6k,
    Gemma4E2bItUqffQ80,
    Gemma4E4bItUqffAfq2,
    Gemma4E4bItUqffAfq3,
    Gemma4E4bItUqffAfq4,
    Gemma4E4bItUqffAfq6,
    Gemma4E4bItUqffAfq8,
    Gemma4E4bItUqffQ2k,
    Gemma4E4bItUqffQ3k,
    Gemma4E4bItUqffQ4k,
    Gemma4E4bItUqffQ5k,
    Gemma4E4bItUqffQ6k,
    Gemma4E4bItUqffQ80,
    Gemma412bItUqffAfq2,
    Gemma412bItUqffAfq3,
    Gemma412bItUqffAfq4,
    Gemma412bItUqffAfq6,
    Gemma412bItUqffAfq8,
    Gemma412bItUqffQ2k,
    Gemma412bItUqffQ3k,
    Gemma412bItUqffQ4k,
    Gemma412bItUqffQ5k,
    Gemma412bItUqffQ6k,
    Gemma412bItUqffQ80,
    Gemma426bA4bItUqffAfq2,
    Gemma426bA4bItUqffAfq3,
    Gemma426bA4bItUqffAfq4,
    Gemma426bA4bItUqffAfq6,
    Gemma426bA4bItUqffAfq8,
    Gemma426bA4bItUqffQ2k,
    Gemma426bA4bItUqffQ3k,
    Gemma426bA4bItUqffQ4k,
    Gemma426bA4bItUqffQ5k,
    Gemma426bA4bItUqffQ6k,
    Gemma426bA4bItUqffQ80,
    Gemma431bItUqffAfq2,
    Gemma431bItUqffAfq3,
    Gemma431bItUqffAfq4,
    Gemma431bItUqffAfq6,
    Gemma431bItUqffAfq8,
    Gemma431bItUqffQ2k,
    Gemma431bItUqffQ3k,
    Gemma431bItUqffQ4k,
    Gemma431bItUqffQ5k,
    Gemma431bItUqffQ6k,
    Gemma431bItUqffQ80,
    Qwen35_27bAppleMetalUqffAfq2,
    Qwen35_27bAppleMetalUqffAfq3,
    Qwen35_27bAppleMetalUqffAfq4,
    Qwen35_27bAppleMetalUqffAfq6,
    Qwen35_27bAppleMetalUqffAfq8,
    Qwen35_27bAppleMetalUqffQ2k,
    Qwen35_27bAppleMetalUqffQ3k,
    Qwen35_27bAppleMetalUqffQ4k,
    Qwen35_27bAppleMetalUqffQ5k,
    Qwen35_27bAppleMetalUqffQ6k,
    Qwen35_27bAppleMetalUqffQ80,
    Qwen35_35bA3bAppleMetalUqffAfq2,
    Qwen35_35bA3bAppleMetalUqffAfq3,
    Qwen35_35bA3bAppleMetalUqffAfq4,
    Qwen35_35bA3bAppleMetalUqffAfq6,
    Qwen35_35bA3bAppleMetalUqffAfq8,
    Qwen35_35bA3bAppleMetalUqffQ2k,
    Qwen35_35bA3bAppleMetalUqffQ3k,
    Qwen35_35bA3bAppleMetalUqffQ4k,
    Qwen35_35bA3bAppleMetalUqffQ5k,
    Qwen35_35bA3bAppleMetalUqffQ6k,
    Qwen35_35bA3bAppleMetalUqffQ80,
    VoxtralMini3b2507AsrStt,
    Dia16bTts,
    Kokoro82mTts,
    Flux2Dev,
    Flux2Schnell,
    Flux2Klein4b,
    Flux2Klein9b,
    EmbeddingGemma300m,
}

impl ModelId {
    /// Returns the family that this model belongs to.
    pub fn family(&self) -> ModelFamily {
        match self {
            ModelId::Gemma4E2bItQ5
            | ModelId::Gemma412bIt
            | ModelId::Gemma426bA4bItQ4
            | ModelId::Gemma4E2bItUqffAfq2
            | ModelId::Gemma4E2bItUqffAfq3
            | ModelId::Gemma4E2bItUqffAfq4
            | ModelId::Gemma4E2bItUqffAfq6
            | ModelId::Gemma4E2bItUqffAfq8
            | ModelId::Gemma4E2bItUqffQ2k
            | ModelId::Gemma4E2bItUqffQ3k
            | ModelId::Gemma4E2bItUqffQ4k
            | ModelId::Gemma4E2bItUqffQ5k
            | ModelId::Gemma4E2bItUqffQ6k
            | ModelId::Gemma4E2bItUqffQ80
            | ModelId::Gemma4E4bItUqffAfq2
            | ModelId::Gemma4E4bItUqffAfq3
            | ModelId::Gemma4E4bItUqffAfq4
            | ModelId::Gemma4E4bItUqffAfq6
            | ModelId::Gemma4E4bItUqffAfq8
            | ModelId::Gemma4E4bItUqffQ2k
            | ModelId::Gemma4E4bItUqffQ3k
            | ModelId::Gemma4E4bItUqffQ4k
            | ModelId::Gemma4E4bItUqffQ5k
            | ModelId::Gemma4E4bItUqffQ6k
            | ModelId::Gemma4E4bItUqffQ80
            | ModelId::Gemma412bItUqffAfq2
            | ModelId::Gemma412bItUqffAfq3
            | ModelId::Gemma412bItUqffAfq4
            | ModelId::Gemma412bItUqffAfq6
            | ModelId::Gemma412bItUqffAfq8
            | ModelId::Gemma412bItUqffQ2k
            | ModelId::Gemma412bItUqffQ3k
            | ModelId::Gemma412bItUqffQ4k
            | ModelId::Gemma412bItUqffQ5k
            | ModelId::Gemma412bItUqffQ6k
            | ModelId::Gemma412bItUqffQ80
            | ModelId::Gemma426bA4bItUqffAfq2
            | ModelId::Gemma426bA4bItUqffAfq3
            | ModelId::Gemma426bA4bItUqffAfq4
            | ModelId::Gemma426bA4bItUqffAfq6
            | ModelId::Gemma426bA4bItUqffAfq8
            | ModelId::Gemma426bA4bItUqffQ2k
            | ModelId::Gemma426bA4bItUqffQ3k
            | ModelId::Gemma426bA4bItUqffQ4k
            | ModelId::Gemma426bA4bItUqffQ5k
            | ModelId::Gemma426bA4bItUqffQ6k
            | ModelId::Gemma426bA4bItUqffQ80
            | ModelId::Gemma431bItUqffAfq2
            | ModelId::Gemma431bItUqffAfq3
            | ModelId::Gemma431bItUqffAfq4
            | ModelId::Gemma431bItUqffAfq6
            | ModelId::Gemma431bItUqffAfq8
            | ModelId::Gemma431bItUqffQ2k
            | ModelId::Gemma431bItUqffQ3k
            | ModelId::Gemma431bItUqffQ4k
            | ModelId::Gemma431bItUqffQ5k
            | ModelId::Gemma431bItUqffQ6k
            | ModelId::Gemma431bItUqffQ80 => ModelFamily::Gemma4,
            ModelId::Qwen35_27bAppleMetalUqffAfq2
            | ModelId::Qwen35_27bAppleMetalUqffAfq3
            | ModelId::Qwen35_27bAppleMetalUqffAfq4
            | ModelId::Qwen35_27bAppleMetalUqffAfq6
            | ModelId::Qwen35_27bAppleMetalUqffAfq8
            | ModelId::Qwen35_27bAppleMetalUqffQ2k
            | ModelId::Qwen35_27bAppleMetalUqffQ3k
            | ModelId::Qwen35_27bAppleMetalUqffQ4k
            | ModelId::Qwen35_27bAppleMetalUqffQ5k
            | ModelId::Qwen35_27bAppleMetalUqffQ6k
            | ModelId::Qwen35_27bAppleMetalUqffQ80
            | ModelId::Qwen35_35bA3bAppleMetalUqffAfq2
            | ModelId::Qwen35_35bA3bAppleMetalUqffAfq3
            | ModelId::Qwen35_35bA3bAppleMetalUqffAfq4
            | ModelId::Qwen35_35bA3bAppleMetalUqffAfq6
            | ModelId::Qwen35_35bA3bAppleMetalUqffAfq8
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ2k
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ3k
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ4k
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ5k
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ6k
            | ModelId::Qwen35_35bA3bAppleMetalUqffQ80 => ModelFamily::Qwen35,
            ModelId::VoxtralMini3b2507AsrStt => ModelFamily::Voxtral,
            ModelId::Dia16bTts => ModelFamily::Dia,
            ModelId::Kokoro82mTts => ModelFamily::Kokoro,
            ModelId::Flux2Dev
            | ModelId::Flux2Schnell
            | ModelId::Flux2Klein4b
            | ModelId::Flux2Klein9b => ModelFamily::Flux2,
            ModelId::EmbeddingGemma300m => ModelFamily::EmbeddingGemma,
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelId::Gemma4E2bItQ5 => f.write_str("gemma-4-e2b-it-q5"),
            ModelId::Gemma412bIt => f.write_str("gemma-4-12b-it"),
            ModelId::Gemma426bA4bItQ4 => f.write_str("gemma-4-26b-a4b-it-q4"),
            ModelId::Gemma4E2bItUqffAfq2 => f.write_str("gemma-4-e2b-it-uqff-afq2"),
            ModelId::Gemma4E2bItUqffAfq3 => f.write_str("gemma-4-e2b-it-uqff-afq3"),
            ModelId::Gemma4E2bItUqffAfq4 => f.write_str("gemma-4-e2b-it-uqff-afq4"),
            ModelId::Gemma4E2bItUqffAfq6 => f.write_str("gemma-4-e2b-it-uqff-afq6"),
            ModelId::Gemma4E2bItUqffAfq8 => f.write_str("gemma-4-e2b-it-uqff-afq8"),
            ModelId::Gemma4E2bItUqffQ2k => f.write_str("gemma-4-e2b-it-uqff-q2k"),
            ModelId::Gemma4E2bItUqffQ3k => f.write_str("gemma-4-e2b-it-uqff-q3k"),
            ModelId::Gemma4E2bItUqffQ4k => f.write_str("gemma-4-e2b-it-uqff-q4k"),
            ModelId::Gemma4E2bItUqffQ5k => f.write_str("gemma-4-e2b-it-uqff-q5k"),
            ModelId::Gemma4E2bItUqffQ6k => f.write_str("gemma-4-e2b-it-uqff-q6k"),
            ModelId::Gemma4E2bItUqffQ80 => f.write_str("gemma-4-e2b-it-uqff-q80"),
            ModelId::Gemma4E4bItUqffAfq2 => f.write_str("gemma-4-e4b-it-uqff-afq2"),
            ModelId::Gemma4E4bItUqffAfq3 => f.write_str("gemma-4-e4b-it-uqff-afq3"),
            ModelId::Gemma4E4bItUqffAfq4 => f.write_str("gemma-4-e4b-it-uqff-afq4"),
            ModelId::Gemma4E4bItUqffAfq6 => f.write_str("gemma-4-e4b-it-uqff-afq6"),
            ModelId::Gemma4E4bItUqffAfq8 => f.write_str("gemma-4-e4b-it-uqff-afq8"),
            ModelId::Gemma4E4bItUqffQ2k => f.write_str("gemma-4-e4b-it-uqff-q2k"),
            ModelId::Gemma4E4bItUqffQ3k => f.write_str("gemma-4-e4b-it-uqff-q3k"),
            ModelId::Gemma4E4bItUqffQ4k => f.write_str("gemma-4-e4b-it-uqff-q4k"),
            ModelId::Gemma4E4bItUqffQ5k => f.write_str("gemma-4-e4b-it-uqff-q5k"),
            ModelId::Gemma4E4bItUqffQ6k => f.write_str("gemma-4-e4b-it-uqff-q6k"),
            ModelId::Gemma4E4bItUqffQ80 => f.write_str("gemma-4-e4b-it-uqff-q80"),
            ModelId::Gemma412bItUqffAfq2 => f.write_str("gemma-4-12b-it-uqff-afq2"),
            ModelId::Gemma412bItUqffAfq3 => f.write_str("gemma-4-12b-it-uqff-afq3"),
            ModelId::Gemma412bItUqffAfq4 => f.write_str("gemma-4-12b-it-uqff-afq4"),
            ModelId::Gemma412bItUqffAfq6 => f.write_str("gemma-4-12b-it-uqff-afq6"),
            ModelId::Gemma412bItUqffAfq8 => f.write_str("gemma-4-12b-it-uqff-afq8"),
            ModelId::Gemma412bItUqffQ2k => f.write_str("gemma-4-12b-it-uqff-q2k"),
            ModelId::Gemma412bItUqffQ3k => f.write_str("gemma-4-12b-it-uqff-q3k"),
            ModelId::Gemma412bItUqffQ4k => f.write_str("gemma-4-12b-it-uqff-q4k"),
            ModelId::Gemma412bItUqffQ5k => f.write_str("gemma-4-12b-it-uqff-q5k"),
            ModelId::Gemma412bItUqffQ6k => f.write_str("gemma-4-12b-it-uqff-q6k"),
            ModelId::Gemma412bItUqffQ80 => f.write_str("gemma-4-12b-it-uqff-q80"),
            ModelId::Gemma426bA4bItUqffAfq2 => f.write_str("gemma-4-26b-a4b-it-uqff-afq2"),
            ModelId::Gemma426bA4bItUqffAfq3 => f.write_str("gemma-4-26b-a4b-it-uqff-afq3"),
            ModelId::Gemma426bA4bItUqffAfq4 => f.write_str("gemma-4-26b-a4b-it-uqff-afq4"),
            ModelId::Gemma426bA4bItUqffAfq6 => f.write_str("gemma-4-26b-a4b-it-uqff-afq6"),
            ModelId::Gemma426bA4bItUqffAfq8 => f.write_str("gemma-4-26b-a4b-it-uqff-afq8"),
            ModelId::Gemma426bA4bItUqffQ2k => f.write_str("gemma-4-26b-a4b-it-uqff-q2k"),
            ModelId::Gemma426bA4bItUqffQ3k => f.write_str("gemma-4-26b-a4b-it-uqff-q3k"),
            ModelId::Gemma426bA4bItUqffQ4k => f.write_str("gemma-4-26b-a4b-it-uqff-q4k"),
            ModelId::Gemma426bA4bItUqffQ5k => f.write_str("gemma-4-26b-a4b-it-uqff-q5k"),
            ModelId::Gemma426bA4bItUqffQ6k => f.write_str("gemma-4-26b-a4b-it-uqff-q6k"),
            ModelId::Gemma426bA4bItUqffQ80 => f.write_str("gemma-4-26b-a4b-it-uqff-q80"),
            ModelId::Gemma431bItUqffAfq2 => f.write_str("gemma-4-31b-it-uqff-afq2"),
            ModelId::Gemma431bItUqffAfq3 => f.write_str("gemma-4-31b-it-uqff-afq3"),
            ModelId::Gemma431bItUqffAfq4 => f.write_str("gemma-4-31b-it-uqff-afq4"),
            ModelId::Gemma431bItUqffAfq6 => f.write_str("gemma-4-31b-it-uqff-afq6"),
            ModelId::Gemma431bItUqffAfq8 => f.write_str("gemma-4-31b-it-uqff-afq8"),
            ModelId::Gemma431bItUqffQ2k => f.write_str("gemma-4-31b-it-uqff-q2k"),
            ModelId::Gemma431bItUqffQ3k => f.write_str("gemma-4-31b-it-uqff-q3k"),
            ModelId::Gemma431bItUqffQ4k => f.write_str("gemma-4-31b-it-uqff-q4k"),
            ModelId::Gemma431bItUqffQ5k => f.write_str("gemma-4-31b-it-uqff-q5k"),
            ModelId::Gemma431bItUqffQ6k => f.write_str("gemma-4-31b-it-uqff-q6k"),
            ModelId::Gemma431bItUqffQ80 => f.write_str("gemma-4-31b-it-uqff-q80"),
            ModelId::Qwen35_27bAppleMetalUqffAfq2 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-afq2")
            }
            ModelId::Qwen35_27bAppleMetalUqffAfq3 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-afq3")
            }
            ModelId::Qwen35_27bAppleMetalUqffAfq4 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-afq4")
            }
            ModelId::Qwen35_27bAppleMetalUqffAfq6 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-afq6")
            }
            ModelId::Qwen35_27bAppleMetalUqffAfq8 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-afq8")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ2k => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q2k")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ3k => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q3k")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ4k => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q4k")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ5k => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q5k")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ6k => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q6k")
            }
            ModelId::Qwen35_27bAppleMetalUqffQ80 => {
                f.write_str("qwen-3.5-27b-apple-metal-uqff-q80")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffAfq2 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-afq2")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffAfq3 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-afq3")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffAfq4 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-afq4")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffAfq6 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-afq6")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffAfq8 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-afq8")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ2k => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q2k")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ3k => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q3k")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ4k => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q4k")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ5k => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q5k")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ6k => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q6k")
            }
            ModelId::Qwen35_35bA3bAppleMetalUqffQ80 => {
                f.write_str("qwen-3.5-35b-a3b-apple-metal-uqff-q80")
            }
            ModelId::VoxtralMini3b2507AsrStt => f.write_str("voxtral-mini-3b-2507-asr-stt"),
            ModelId::Dia16bTts => f.write_str("dia-16b-tts"),
            ModelId::Kokoro82mTts => f.write_str("kokoro-82m-tts"),
            ModelId::Flux2Dev => f.write_str("flux-2-dev"),
            ModelId::Flux2Schnell => f.write_str("flux-2-schnell"),
            ModelId::Flux2Klein4b => f.write_str("flux-2-klein-4b"),
            ModelId::Flux2Klein9b => f.write_str("flux-2-klein-9b"),
            ModelId::EmbeddingGemma300m => f.write_str("embedding-gemma-300m"),
        }
    }
}
