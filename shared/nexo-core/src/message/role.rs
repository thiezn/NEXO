use serde::{Deserialize, Serialize};

/// The semantic role of a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// System-level instructions and global context.
    System,

    /// Developer-authored instructions that are distinct from user input.
    Developer,

    /// End-user input.
    User,

    /// Model-authored output.
    Assistant,

    /// Tool-originated output injected back into the conversation.
    Tool,
}
