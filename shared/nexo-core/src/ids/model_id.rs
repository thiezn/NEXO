use crate::ModelFamily;
use serde::{Deserialize, Serialize};
use strum::ParseError;

/// A stable identifier for an inference model.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    Copy,
)]
#[serde(into = "String", try_from = "String")]
#[allow(missing_docs)]
pub enum ModelId {
    // Gemma4E2bItQ5,
    // Gemma412bIt,
    // Gemma426bA4bItQ4,
    // Gemma4E2bItUqffAfq2,
    // Gemma4E2bItUqffAfq3,
    // Gemma4E2bItUqffAfq4,
    // Gemma4E2bItUqffAfq6,
    // Gemma4E2bItUqffAfq8,
    // Gemma4E4bItUqffAfq2,
    // Gemma4E4bItUqffAfq3,
    // Gemma4E4bItUqffAfq4,
    #[strum(serialize = "gemma-4-e4b-it-uqff-afq6")]
    Gemma4E4bItUqffAfq6,
    #[strum(serialize = "gemma-4-e4b-it-uqff-afq8")]
    Gemma4E4bItUqffAfq8,
    // Gemma412bItUqffAfq2,
    // Gemma412bItUqffAfq3,
    // Gemma412bItUqffAfq4,
    // Gemma412bItUqffAfq6,
    // Gemma412bItUqffAfq8,
    // Gemma426bA4bItUqffAfq2,
    // Gemma426bA4bItUqffAfq3,
    // Gemma426bA4bItUqffAfq4,
    #[strum(serialize = "gemma-4-26b-a4b-it-uqff-afq6")]
    Gemma426bA4bItUqffAfq6,
    #[strum(serialize = "gemma-4-26b-a4b-it-uqff-afq8")]
    Gemma426bA4bItUqffAfq8,
    // Gemma431bItUqffAfq2,
    // Gemma431bItUqffAfq3,
    // Gemma431bItUqffAfq4,
    // Gemma431bItUqffAfq6,
    // Gemma431bItUqffAfq8,
    // Qwen35_27bAppleMetalUqffAfq2,
    // Qwen35_27bAppleMetalUqffAfq3,
    // Qwen35_27bAppleMetalUqffAfq4,
    // Qwen35_27bAppleMetalUqffAfq6,
    // Qwen35_27bAppleMetalUqffAfq8,
    // Qwen35_35bA3bAppleMetalUqffAfq2,
    // Qwen35_35bA3bAppleMetalUqffAfq3,
    // Qwen35_35bA3bAppleMetalUqffAfq4,
    // Qwen35_35bA3bAppleMetalUqffAfq6,
    // Qwen35_35bA3bAppleMetalUqffAfq8,
    // Qwen35_35bA3bAppleMetalUqffQ2k,
    // Qwen35_35bA3bAppleMetalUqffQ3k,
    // Qwen35_35bA3bAppleMetalUqffQ4k,
    // Qwen35_35bA3bAppleMetalUqffQ5k,
    // Qwen35_35bA3bAppleMetalUqffQ6k,
    // Qwen35_35bA3bAppleMetalUqffQ80,
    // VoxtralMini3b2507AsrStt,
    #[strum(serialize = "kokoro-82m")]
    Kokoro82m,
    // Flux2Dev,
    // Flux2Schnell,
    // Flux2Klein4b,
    #[strum(serialize = "flux-2-klein-9b")]
    Flux2Klein9b,
    #[strum(serialize = "embedding-gemma-300m")]
    EmbeddingGemma300m,
}

impl ModelId {
    /// Returns the family that this model belongs to.
    pub fn family(&self) -> ModelFamily {
        match self {
            // ModelId::Gemma4E2bItQ5
            // | ModelId::Gemma412bIt
            // | ModelId::Gemma426bA4bItQ4
            // | ModelId::Gemma4E2bItUqffAfq2
            // | ModelId::Gemma4E2bItUqffAfq3
            // | ModelId::Gemma4E2bItUqffAfq4
            // | ModelId::Gemma4E2bItUqffAfq6
            // | ModelId::Gemma4E2bItUqffAfq8
            // | ModelId::Gemma4E4bItUqffAfq2
            // | ModelId::Gemma4E4bItUqffAfq3
            // | ModelId::Gemma4E4bItUqffAfq4
            | ModelId::Gemma4E4bItUqffAfq6
            | ModelId::Gemma4E4bItUqffAfq8
            // | ModelId::Gemma412bItUqffAfq2
            // | ModelId::Gemma412bItUqffAfq3
            // | ModelId::Gemma412bItUqffAfq4
            // | ModelId::Gemma412bItUqffAfq6
            // | ModelId::Gemma412bItUqffAfq8
            // | ModelId::Gemma426bA4bItUqffAfq2
            // | ModelId::Gemma426bA4bItUqffAfq3
            // | ModelId::Gemma426bA4bItUqffAfq4
            | ModelId::Gemma426bA4bItUqffAfq6
            | ModelId::Gemma426bA4bItUqffAfq8=> ModelFamily::Gemma4,
            // | ModelId::Gemma426bA4bItUqffQ2k
            // | ModelId::Gemma426bA4bItUqffQ3k
            // | ModelId::Gemma426bA4bItUqffQ4k
            // | ModelId::Gemma426bA4bItUqffQ5k
            // | ModelId::Gemma426bA4bItUqffQ6k
            // | ModelId::Gemma431bItUqffAfq2
            // | ModelId::Gemma431bItUqffAfq3
            // | ModelId::Gemma431bItUqffAfq4
            // | ModelId::Gemma431bItUqffAfq6
            // | ModelId::Gemma431bItUqffAfq8
            // | ModelId::Gemma431bItUqffQ80 => ModelFamily::Gemma4,
            // ModelId::Qwen35_27bAppleMetalUqffAfq2
            // | ModelId::Qwen35_27bAppleMetalUqffAfq3
            // | ModelId::Qwen35_27bAppleMetalUqffAfq4
            // | ModelId::Qwen35_27bAppleMetalUqffAfq6
            // | ModelId::Qwen35_27bAppleMetalUqffAfq8
            // | ModelId::Qwen35_35bA3bAppleMetalUqffAfq2
            // | ModelId::Qwen35_35bA3bAppleMetalUqffAfq3
            // | ModelId::Qwen35_35bA3bAppleMetalUqffAfq4
            // | ModelId::Qwen35_35bA3bAppleMetalUqffAfq6
            // | ModelId::Qwen35_35bA3bAppleMetalUqffAfq8
            // | ModelId::Qwen35_35bA3bAppleMetalUqffQ80 => ModelFamily::Qwen35,
            // ModelId::VoxtralMini3b2507AsrStt => ModelFamily::Voxtral,
            ModelId::Kokoro82m => ModelFamily::Kokoro,
            // ModelId::Flux2Dev
            // | ModelId::Flux2Schnell
            // | ModelId::Flux2Klein4b
            | ModelId::Flux2Klein9b => ModelFamily::Flux2,
            ModelId::EmbeddingGemma300m => ModelFamily::EmbeddingGemma,
        }
    }
}

impl From<ModelId> for String {
    fn from(model_id: ModelId) -> Self {
        model_id.to_string()
    }
}

impl TryFrom<String> for ModelId {
    type Error = ParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl PartialEq<str> for ModelId {
    fn eq(&self, other: &str) -> bool {
        self.as_ref() == other
    }
}

impl PartialEq<ModelId> for str {
    fn eq(&self, other: &ModelId) -> bool {
        other == self
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::ModelId;

    #[test]
    fn serializes_to_model_id_strings() {
        assert_eq!(
            serde_json::to_string(&ModelId::Gemma4E4bItUqffAfq8).unwrap(),
            "\"gemma-4-e4b-it-uqff-afq8\""
        );
        assert_eq!(
            serde_json::to_string(&ModelId::Kokoro82m).unwrap(),
            "\"kokoro-82m\""
        );
        assert_eq!(
            serde_json::to_string(&ModelId::Flux2Klein9b).unwrap(),
            "\"flux-2-klein-9b\""
        );
    }

    #[test]
    fn deserializes_to_model_ids() {
        assert_eq!(
            serde_json::from_str::<ModelId>("\"gemma-4-e4b-it-uqff-afq8\"").unwrap(),
            ModelId::Gemma4E4bItUqffAfq8
        );
        assert_eq!(
            serde_json::from_str::<ModelId>("\"kokoro-82m\"").unwrap(),
            ModelId::Kokoro82m
        );
        assert_eq!(
            serde_json::from_str::<ModelId>("\"flux-2-klein-9b\"").unwrap(),
            ModelId::Flux2Klein9b
        );
    }
}
