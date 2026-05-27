use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Category of LoRA adapters for image generation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageLoraCategory {
    HeroImage,
    BackgroundImage,
    Object,
    Style,
}

/// Category of LoRA adapters for tool-calling models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolLoraCategory {
    ToolCalling,
}

/// A LoRA adapter that can be applied on top of a base model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub id: String,
    pub name: String,
    pub base_model_id: String,
    pub weights_path: PathBuf,
    pub config_path: PathBuf,
    pub trigger_words: Vec<String>,
    pub default_strength: f32,
}

/// Trait for models that support LoRA adapter hot-swapping.
pub trait LoraCapable<C> {
    /// Apply a LoRA adapter with the given strength (0.0 – 1.0).
    fn apply_lora(&mut self, adapter: &LoraAdapter, strength: f32) -> Result<()>;

    /// Remove the currently active LoRA adapter.
    fn remove_lora(&mut self) -> Result<()>;

    /// Return the currently active LoRA adapter, if any.
    fn active_lora(&self) -> Option<&LoraAdapter>;

    /// Return the LoRA categories this model supports.
    fn supported_lora_categories(&self) -> &[C];
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn lora_adapter_serde_roundtrip() {
        let adapter = LoraAdapter {
            id: "lora-001".into(),
            name: "Pixel Art Style".into(),
            base_model_id: "flux-schnell".into(),
            weights_path: PathBuf::from("/models/loras/pixel_art.safetensors"),
            config_path: PathBuf::from("/models/loras/pixel_art.json"),
            trigger_words: vec!["pixel art".into(), "pixelated".into()],
            default_strength: 0.8,
        };

        let json = serde_json::to_string_pretty(&adapter).unwrap();
        let parsed: LoraAdapter = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "lora-001");
        assert_eq!(parsed.name, "Pixel Art Style");
        assert_eq!(parsed.base_model_id, "flux-schnell");
        assert_eq!(
            parsed.weights_path,
            PathBuf::from("/models/loras/pixel_art.safetensors")
        );
        assert_eq!(
            parsed.config_path,
            PathBuf::from("/models/loras/pixel_art.json")
        );
        assert_eq!(parsed.trigger_words, vec!["pixel art", "pixelated"]);
        assert!((parsed.default_strength - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn image_lora_category_serde_roundtrip() {
        let categories = [
            ImageLoraCategory::HeroImage,
            ImageLoraCategory::BackgroundImage,
            ImageLoraCategory::Object,
            ImageLoraCategory::Style,
        ];

        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: ImageLoraCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn image_lora_category_serializes_to_snake_case() {
        assert_eq!(
            serde_json::to_string(&ImageLoraCategory::HeroImage).unwrap(),
            "\"hero_image\""
        );
        assert_eq!(
            serde_json::to_string(&ImageLoraCategory::BackgroundImage).unwrap(),
            "\"background_image\""
        );
        assert_eq!(
            serde_json::to_string(&ImageLoraCategory::Object).unwrap(),
            "\"object\""
        );
        assert_eq!(
            serde_json::to_string(&ImageLoraCategory::Style).unwrap(),
            "\"style\""
        );
    }

    #[test]
    fn tool_lora_category_serde_roundtrip() {
        let cat = ToolLoraCategory::ToolCalling;
        let json = serde_json::to_string(&cat).unwrap();
        let parsed: ToolLoraCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(cat, parsed);
    }

    #[test]
    fn tool_lora_category_serializes_to_snake_case() {
        assert_eq!(
            serde_json::to_string(&ToolLoraCategory::ToolCalling).unwrap(),
            "\"tool_calling\""
        );
    }
}
