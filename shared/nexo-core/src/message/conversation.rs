use serde::{Deserialize, Serialize};

use super::{ContentPart, MessageRole};

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConversationMessage {
    /// The semantic author role of the message.
    role: MessageRole,

    /// The ordered content parts carried by the message.
    parts: Vec<ContentPart>,
}

impl ConversationMessage {
    /// Creates a new conversation message with the given role and content.
    pub fn new(role: impl Into<MessageRole>, content: Vec<ContentPart>) -> Self {
        Self {
            role: role.into(),
            parts: content,
        }
    }

    /// Creates a new user message with the given text content.
    pub fn new_text(text: &str) -> Self {
        Self {
            role: MessageRole::User,
            parts: vec![ContentPart::Text(text.to_string())],
        }
    }

    /// Creates a new system prompt message with the given content.
    ///
    /// # Arguments
    ///
    /// content - The content of the system prompt message.
    pub fn new_system_prompt(content: &str) -> Self {
        Self {
            role: MessageRole::System,
            parts: vec![ContentPart::Text(content.to_string())],
        }
    }

    /// Creates a new developer prompt message with the given content.
    ///
    /// # Arguments
    ///
    /// * `content` - The content of the developer prompt message.
    pub fn new_developer_prompt(content: &str) -> Self {
        Self {
            role: MessageRole::Developer,
            parts: vec![ContentPart::Text(content.to_string())],
        }
    }

    /// Returns the role of the message.
    pub fn role(&self) -> &MessageRole {
        &self.role
    }

    /// Returns the content parts of the message.
    pub fn parts(&self) -> &Vec<ContentPart> {
        &self.parts
    }
}

/// The ordered message history submitted to a model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Conversation {
    /// The ordered messages that make up the conversation.
    pub messages: Vec<ConversationMessage>,
}
