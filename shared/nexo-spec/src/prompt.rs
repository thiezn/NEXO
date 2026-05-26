use serde::{Deserialize, Serialize};

/// A stored prompt document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct PromptDocument {
    pub id: String,
    pub content: String,
}

/// An ordered collection of prompt documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct PromptCollection {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub documents: Vec<String>,
}

/// The assembled system prompt sent to a model round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SystemPrompt {
    pub content: String,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn prompt_collection_serializes_with_documents() {
        let collection = PromptCollection {
            id: "default".into(),
            name: "Default".into(),
            description: Some("Core system identity".into()),
            documents: vec!["identity.md".into(), "skills.md".into()],
        };

        let json = serde_json::to_value(&collection).unwrap();

        assert_eq!(json["id"], "default");
        assert_eq!(json["documents"][0], "identity.md");
        assert_eq!(json["documents"][1], "skills.md");
    }

    #[test]
    fn system_prompt_serializes() {
        let prompt = SystemPrompt {
            content: "You are helpful.".into(),
        };

        let json = serde_json::to_value(&prompt).unwrap();
        assert_eq!(json["content"], "You are helpful.");
    }
}
