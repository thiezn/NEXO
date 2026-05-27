use serde::{Deserialize, Serialize};

use crate::ids::ModelId;

use super::ModelCapability;

/// Describes how a caller wants a model to be selected.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelSelection {
    /// The exact model to use, if the caller requires one.
    pub specific_model: Option<ModelId>,

    /// Capabilities that must be present on the selected model.
    pub required_capabilities: Vec<ModelCapability>,

    /// Capabilities that should be preferred when multiple models qualify.
    pub preferred_capabilities: Vec<ModelCapability>,
}
