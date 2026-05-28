use serde::{Deserialize, Serialize};

use crate::common::MetadataMap;

use super::{ContentPart, MessageRole};

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConversationMessage {
    /// The semantic author role of the message.
    pub role: MessageRole,

    /// The ordered content parts carried by the message.
    pub parts: Vec<ContentPart>,

    /// Arbitrary metadata attached to the message.
    pub metadata: MetadataMap,
}

/// The ordered message history submitted to a model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Conversation {
    /// The ordered messages that make up the conversation.
    pub messages: Vec<ConversationMessage>,

    /// Arbitrary metadata attached to the conversation.
    pub metadata: MetadataMap,
}
