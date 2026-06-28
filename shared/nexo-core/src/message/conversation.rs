use serde::{Deserialize, Serialize};

use super::{ContentPart, MessageRole};

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConversationMessage {
    /// The semantic author role of the message.
    pub role: MessageRole,

    /// The ordered content parts carried by the message.
    pub parts: Vec<ContentPart>,
}

/// The ordered message history submitted to a model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Conversation {
    /// The ordered messages that make up the conversation.
    pub messages: Vec<ConversationMessage>,
}
