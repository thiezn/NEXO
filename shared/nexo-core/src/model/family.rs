use serde::{Deserialize, Serialize};

/// The family that a specific model belongs to.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[allow(missing_docs)]
pub enum ModelFamily {
    EmbeddingGemma,
    Gemma4,
    Qwen35,
    Voxtral,
    Kokoro,
    Flux2,
}
