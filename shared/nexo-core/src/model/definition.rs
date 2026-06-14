use serde::{Deserialize, Serialize};

use crate::common::MetadataMap;
use crate::ids::ModelId;

use super::{ModelCapability, RoleStrategy};

/// Describes a model that may be selected for inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelDefinition {
    /// The stable model identifier
    pub id: ModelId,

    /// The human-readable model label
    pub display_name: String,

    /// The declared model capabilities
    pub capabilities: Vec<ModelCapability>,

    /// The conversation role handling strategy required by the model adapter.
    pub role_strategy: RoleStrategy,

    /// The maximum context window, measured in tokens, if known.
    pub context_window_tokens: Option<usize>,

    /// The maximum output token budget, if the model exposes one.
    pub max_output_tokens: Option<usize>,

    /// Additional provider-specific metadata.
    pub metadata: MetadataMap,
}
