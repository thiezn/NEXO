use serde::{Deserialize, Serialize};

/// The family that a specific model belongs to.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
pub enum ModelFamily {
    /// The family of models that are based on the Gemma architecture.
    EmbeddingGemma,

    /// The family of models that are based on the Gemma4 architecture.
    Gemma4,

    /// The family of models that are based on the Qwen35 architecture.
    Qwen35,

    /// The family of models that are based on the Voxtral architecture.
    Voxtral,

    /// The family of models that are based on the Kokoro architecture.
    Kokoro,

    /// The family of models that are based on the Flux2 architecture.
    Flux2,
}
