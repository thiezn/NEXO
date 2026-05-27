use serde::{Deserialize, Serialize};

/// A modality accepted as input or produced as output by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SupportedModality {
    /// Plain textual content.
    Text,

    /// Still image content.
    Image,

    /// Video content.
    Video,

    /// Audio content.
    Audio,
}

/// Declares which modalities a model accepts and produces.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelModalities {
    /// The modalities accepted by the model as request input.
    pub input: Vec<SupportedModality>,

    /// The modalities produced by the model as response output.
    pub output: Vec<SupportedModality>,
}
