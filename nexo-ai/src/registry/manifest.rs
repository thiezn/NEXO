use crate::download::{Component, ModelManifest};
use crate::shared::types::ModelCategory;
use std::sync::LazyLock;

/// Component types for AI model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiComponent {
    Model,
    ModelShard,
    Tokenizer,
    Config,
    Vae,
    TextEncoder,
    ClipEncoder,
    T5Encoder,
}

impl Component for AiComponent {
    fn name(&self) -> &str {
        match self {
            Self::Model => "model",
            Self::ModelShard => "model_shard",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::ClipEncoder => "clip_encoder",
            Self::T5Encoder => "t5_encoder",
        }
    }

    fn is_model_specific(&self) -> bool {
        match self {
            Self::Model | Self::ModelShard | Self::Tokenizer | Self::Config => true,
            Self::Vae | Self::TextEncoder | Self::ClipEncoder | Self::T5Encoder => false,
        }
    }
}

/// An AI model manifest with associated categories.
pub struct AiModelManifest {
    pub manifest: ModelManifest<AiComponent>,
    pub categories: Vec<ModelCategory>,
}

// ── Registry ────────────────────────────────────────────────────────────────

fn build_all_manifests() -> Vec<AiModelManifest> {
    // Start empty -- manifests are added as models are integrated.
    vec![]
}

static ALL_MANIFESTS: LazyLock<Vec<AiModelManifest>> = LazyLock::new(build_all_manifests);

/// Return all known AI model manifests.
pub fn known_manifests() -> &'static [AiModelManifest] {
    &ALL_MANIFESTS
}

/// Look up a manifest by name (case-sensitive).
pub fn find_manifest(name: &str) -> Option<&'static AiModelManifest> {
    ALL_MANIFESTS.iter().find(|m| m.manifest.name == name)
}

/// Return all manifests that belong to a given category.
pub fn manifests_for_category(category: ModelCategory) -> Vec<&'static AiModelManifest> {
    ALL_MANIFESTS
        .iter()
        .filter(|m| m.categories.contains(&category))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ai_component_names() {
        assert_eq!(AiComponent::Model.name(), "model");
        assert_eq!(AiComponent::ModelShard.name(), "model_shard");
        assert_eq!(AiComponent::Tokenizer.name(), "tokenizer");
        assert_eq!(AiComponent::Config.name(), "config");
        assert_eq!(AiComponent::Vae.name(), "vae");
        assert_eq!(AiComponent::TextEncoder.name(), "text_encoder");
        assert_eq!(AiComponent::ClipEncoder.name(), "clip_encoder");
        assert_eq!(AiComponent::T5Encoder.name(), "t5_encoder");
    }

    #[test]
    fn ai_component_model_specificity() {
        assert!(AiComponent::Model.is_model_specific());
        assert!(AiComponent::ModelShard.is_model_specific());
        assert!(AiComponent::Tokenizer.is_model_specific());
        assert!(AiComponent::Config.is_model_specific());

        assert!(!AiComponent::Vae.is_model_specific());
        assert!(!AiComponent::TextEncoder.is_model_specific());
        assert!(!AiComponent::ClipEncoder.is_model_specific());
        assert!(!AiComponent::T5Encoder.is_model_specific());
    }

    #[test]
    fn known_manifests_starts_empty() {
        assert!(known_manifests().is_empty());
    }

    #[test]
    fn find_manifest_returns_none_for_unknown() {
        assert!(find_manifest("nonexistent-model").is_none());
    }

    #[test]
    fn manifests_for_category_returns_empty() {
        assert!(manifests_for_category(ModelCategory::Chat).is_empty());
    }
}
