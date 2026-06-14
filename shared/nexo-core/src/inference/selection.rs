use crate::{ModelCapability, ModelId};
use serde::{Deserialize, Serialize};

/// Criteria for selecting a model to fulfill an inference request,
/// either by specific identifier or by required capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ModelSelection {
    /// Select a specific model by its unique identifier.
    SpecificModel(ModelId),

    /// Select any model that supports the required capabilities, optionally preferring those that also support preferred capabilities.
    Capabilities(Vec<ModelCapability>),
}
