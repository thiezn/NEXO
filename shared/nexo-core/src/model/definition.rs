use crate::{MetadataMap, ModelCapability, ModelId, RoleStrategy};
use serde::{Deserialize, Serialize};

/// Describes a model that may be selected for inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelDefinition {
    /// The stable model identifier
    id: ModelId,

    /// The human-readable model label
    display_name: String,

    /// The declared model capabilities
    capabilities: Vec<ModelCapability>,

    /// The conversation role handling strategy required by the model adapter.
    role_strategy: RoleStrategy,

    /// The maximum context window, measured in tokens, if known.
    context_window_tokens: Option<usize>,

    /// The maximum output token budget, if the model exposes one.
    max_output_tokens: Option<usize>,

    /// Additional provider-specific metadata.
    metadata: MetadataMap,
}

impl ModelDefinition {
    /// The stable model identifier.
    pub fn id(&self) -> &ModelId {
        &self.id
    }

    /// The human-readable model label.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// The declared model capabilities.
    pub fn capabilities(&self) -> &[ModelCapability] {
        &self.capabilities
    }

    /// The conversation role handling strategy required by the model adapter.  
    pub fn role_strategy(&self) -> &RoleStrategy {
        &self.role_strategy
    }

    /// The maximum context window, measured in tokens, if known.
    pub fn context_window_tokens(&self) -> Option<usize> {
        self.context_window_tokens
    }

    /// The maximum output token budget, if the model exposes one.
    pub fn max_output_tokens(&self) -> Option<usize> {
        self.max_output_tokens
    }

    /// Additional provider-specific metadata.
    pub fn metadata(&self) -> &MetadataMap {
        &self.metadata
    }

    /// Initialize a new ModelDefinition for a given model ID.
    ///
    ///
    pub fn new(model_id: ModelId) -> Self {
        match model_id {
            ModelId::Gemma4E4bItUqffQ80 => Self {
                id: model_id,
                display_name: "Gemma 4 E4B IT Q8.0".to_string(),
                capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::AudioInput,
                    ModelCapability::ImageInput,
                    ModelCapability::VideoInput,
                    ModelCapability::Reasoning,
                    ModelCapability::StructuredOutput,
                    ModelCapability::Streaming,
                ],
                role_strategy: RoleStrategy::Default,
                context_window_tokens: Some(32768),
                max_output_tokens: Some(8192),
                metadata: MetadataMap::new(),
            },
            ModelId::Gemma426bA4bItUqffQ80 => Self {
                id: model_id,
                display_name: "Gemma 4 26B A4B IT Q8.0".to_string(),
                capabilities: vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::ToolCalling,
                    ModelCapability::ImageInput,
                    ModelCapability::VideoInput,
                    ModelCapability::Reasoning,
                    ModelCapability::StructuredOutput,
                    ModelCapability::Streaming,
                ],
                role_strategy: RoleStrategy::Default,
                context_window_tokens: Some(32768),
                max_output_tokens: Some(8192),
                metadata: MetadataMap::new(),
            },
            ModelId::Kokoro82m => Self {
                id: model_id,
                display_name: "Kokoro 82M".to_string(),
                capabilities: vec![ModelCapability::SpeechGeneration],
                role_strategy: RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: MetadataMap::new(),
            },
            ModelId::EmbeddingGemma300m => Self {
                id: model_id,
                display_name: "Embedding Gemma 300M".to_string(),
                capabilities: vec![ModelCapability::Embeddings],
                role_strategy: RoleStrategy::Default,
                context_window_tokens: None,
                max_output_tokens: None,
                metadata: MetadataMap::new(),
            },
            ModelId::Flux2Klein9b => Self {
                id: model_id,
                display_name: "Flux 2 Klein 9B".to_string(),
                capabilities: vec![ModelCapability::ImageGeneration],
                role_strategy: RoleStrategy::Default,
                context_window_tokens: Some(32768),
                max_output_tokens: Some(8192),
                metadata: MetadataMap::new(),
            },
        }
    }
}
