use serde::{Deserialize, Serialize};

use crate::common::MetadataMap;
use crate::ids::ModelId;

use super::{InferenceRuntime, ModelCapability, ModelModalities, RoleStrategy};

/// Describes a model that may be selected for inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelDescriptor {
    /// The stable model identifier used across the workspace.
    pub id: ModelId,

    /// The human-readable model label shown in logs or user interfaces.
    pub display_name: String,

    /// The provider or runtime family responsible for the model.
    pub provider: Option<String>,

    /// The runtime implementation required or preferred when loading this model.
    #[serde(default)]
    pub runtime: InferenceRuntime,

    /// The declared model capabilities.
    pub capabilities: Vec<ModelCapability>,

    /// The modalities accepted and emitted by the model.
    pub modalities: ModelModalities,

    /// The conversation role handling strategy required by the model adapter.
    pub role_strategy: RoleStrategy,

    /// The maximum context window, measured in tokens, if known.
    pub context_window_tokens: Option<usize>,

    /// The maximum output token budget, if the model exposes one.
    pub max_output_tokens: Option<usize>,

    /// Additional provider-specific metadata.
    pub metadata: MetadataMap,
}
