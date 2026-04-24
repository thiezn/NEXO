use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The category of capability a model provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    Chat,
    Tool,
    Image,
    Listen,
    Talk,
    Imagine,
    Embed,
}

impl ModelCategory {
    /// Return the kebab-case string for this category.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Tool => "tool",
            Self::Image => "image",
            Self::Listen => "listen",
            Self::Talk => "talk",
            Self::Imagine => "imagine",
            Self::Embed => "embed",
        }
    }

    /// All variants in declaration order.
    pub fn all() -> &'static [ModelCategory] {
        &[
            Self::Chat,
            Self::Tool,
            Self::Image,
            Self::Listen,
            Self::Talk,
            Self::Imagine,
            Self::Embed,
        ]
    }
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModelCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat" => Ok(Self::Chat),
            "tool" => Ok(Self::Tool),
            "image" => Ok(Self::Image),
            "listen" => Ok(Self::Listen),
            "talk" => Ok(Self::Talk),
            "imagine" => Ok(Self::Imagine),
            "embed" => Ok(Self::Embed),
            other => Err(format!("unknown model category: {other}")),
        }
    }
}

/// A loaded model with its supported categories.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct LoadedModelInfo {
    pub model_id: String,
    pub categories: Vec<ModelCategory>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn model_category_serde_roundtrip() {
        for &cat in ModelCategory::all() {
            let json = serde_json::to_string(&cat).unwrap();
            let parsed: ModelCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn model_category_serializes_to_snake_case() {
        assert_eq!(
            serde_json::to_string(&ModelCategory::Chat).unwrap(),
            "\"chat\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Tool).unwrap(),
            "\"tool\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Image).unwrap(),
            "\"image\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Listen).unwrap(),
            "\"listen\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Talk).unwrap(),
            "\"talk\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Imagine).unwrap(),
            "\"imagine\""
        );
        assert_eq!(
            serde_json::to_string(&ModelCategory::Embed).unwrap(),
            "\"embed\""
        );
    }

    #[test]
    fn model_category_all_is_complete() {
        let all = ModelCategory::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn model_category_as_str() {
        assert_eq!(ModelCategory::Chat.as_str(), "chat");
        assert_eq!(ModelCategory::Imagine.as_str(), "imagine");
    }

    #[test]
    fn model_category_display() {
        for &cat in ModelCategory::all() {
            assert_eq!(format!("{cat}"), cat.as_str());
        }
    }

    #[test]
    fn model_category_from_str_valid() {
        for &cat in ModelCategory::all() {
            let parsed: ModelCategory = cat.as_str().parse().unwrap();
            assert_eq!(cat, parsed);
        }
    }

    #[test]
    fn model_category_from_str_invalid() {
        let result: Result<ModelCategory, _> = "nonexistent".parse();
        assert!(result.is_err());
    }

    #[test]
    fn loaded_model_info_serde_roundtrip() {
        let info = LoadedModelInfo {
            model_id: "gemma-4-e4b-it".into(),
            categories: vec![
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image,
            ],
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: LoadedModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, parsed);
    }

    #[test]
    fn loaded_model_info_camel_case() {
        let info = LoadedModelInfo {
            model_id: "test".into(),
            categories: vec![ModelCategory::Chat],
        };
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["modelId"], "test");
        assert_eq!(json["categories"][0], "chat");
    }
}
