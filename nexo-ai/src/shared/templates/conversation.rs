use crate::shared::types::{ChatMessage, ChatRole};

use super::{ChatTemplate, ReasoningMode};

/// Manages conversation history and prompt formatting.
///
/// Model-agnostic: stores messages and delegates formatting to a `ChatTemplate`.
/// Designed for future extension to persistent storage (sqlite/file).
pub struct ConversationManager {
    messages: Vec<ChatMessage>,
    system_prompt: Option<String>,
    reasoning: ReasoningMode,
}

impl ConversationManager {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
            reasoning: ReasoningMode::default(),
        }
    }

    pub fn with_system_prompt(prompt: String) -> Self {
        let content = prompt.clone();
        Self {
            messages: vec![ChatMessage {
                role: ChatRole::System,
                content,
            }],
            system_prompt: Some(prompt),
            reasoning: ReasoningMode::default(),
        }
    }

    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    /// Get all messages (including system prompt if set).
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Clear conversation history, keeping the system prompt if present.
    pub fn clear(&mut self) {
        self.messages.clear();
        if let Some(ref prompt) = self.system_prompt {
            self.messages.push(ChatMessage {
                role: ChatRole::System,
                content: prompt.clone(),
            });
        }
    }

    /// Apply sliding window: keep system prompt + last N user/assistant turn pairs.
    pub fn slide(&mut self, keep_turns: usize) {
        let system_msg = if self.system_prompt.is_some() {
            self.messages.first().cloned()
        } else {
            None
        };

        // Count turn pairs (user + assistant = 1 turn).
        let non_system: Vec<ChatMessage> = self
            .messages
            .iter()
            .filter(|m| m.role != ChatRole::System)
            .cloned()
            .collect();

        // Keep the last `keep_turns * 2` non-system messages (user + assistant pairs).
        let keep_count = keep_turns * 2;
        let start = non_system.len().saturating_sub(keep_count);
        let kept = &non_system[start..];

        self.messages.clear();
        if let Some(sys) = system_msg {
            self.messages.push(sys);
        }
        self.messages.extend_from_slice(kept);
    }

    /// Replace older messages with a summary, preserving system prompt.
    /// The summary becomes a new system-level context message.
    pub fn summarize_prefix(&mut self, summary: String) {
        let system_msg = if self.system_prompt.is_some() {
            self.messages.first().cloned()
        } else {
            None
        };

        let len = self.messages.len();
        let start = len.saturating_sub(2);
        let last_pair: Vec<ChatMessage> = self.messages[start..].to_vec();

        self.messages.clear();
        if let Some(sys) = system_msg {
            self.messages.push(sys);
        }
        self.messages.push(ChatMessage {
            role: ChatRole::System,
            content: format!("Previous conversation summary: {summary}"),
        });
        self.messages.extend(last_pair);
    }

    /// Number of user/assistant turn pairs.
    pub fn turn_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| m.role == ChatRole::User)
            .count()
    }

    /// Format the full conversation using the given template.
    pub fn format(&self, template: &dyn ChatTemplate) -> String {
        template.format_prompt(&self.messages, &self.reasoning)
    }

    /// Format with tools using the given template.
    pub fn format_with_tools(
        &self,
        template: &dyn ChatTemplate,
        tools: &[nexo_tool_spec::tool::ToolSpec],
    ) -> String {
        template.format_with_tools(&self.messages, tools, &self.reasoning)
    }

    pub fn set_reasoning(&mut self, mode: ReasoningMode) {
        self.reasoning = mode;
    }

    pub fn reasoning(&self) -> &ReasoningMode {
        &self.reasoning
    }
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let cm = ConversationManager::new();
        assert!(cm.messages().is_empty());
        assert_eq!(cm.turn_count(), 0);
    }

    #[test]
    fn with_system_prompt_starts_with_system() {
        let cm = ConversationManager::with_system_prompt("Be helpful.".into());
        assert_eq!(cm.messages().len(), 1);
        assert_eq!(cm.messages()[0].role, ChatRole::System);
        assert_eq!(cm.messages()[0].content, "Be helpful.");
    }

    #[test]
    fn push_and_turn_count() {
        let mut cm = ConversationManager::new();
        cm.push(ChatMessage {
            role: ChatRole::User,
            content: "hi".into(),
        });
        cm.push(ChatMessage {
            role: ChatRole::Assistant,
            content: "hello".into(),
        });
        assert_eq!(cm.turn_count(), 1);
        assert_eq!(cm.messages().len(), 2);
    }

    #[test]
    fn clear_keeps_system_prompt() {
        let mut cm = ConversationManager::with_system_prompt("system".into());
        cm.push(ChatMessage {
            role: ChatRole::User,
            content: "hi".into(),
        });
        assert_eq!(cm.messages().len(), 2);
        cm.clear();
        assert_eq!(cm.messages().len(), 1);
        assert_eq!(cm.messages()[0].role, ChatRole::System);
    }

    #[test]
    fn clear_empty_without_system() {
        let mut cm = ConversationManager::new();
        cm.push(ChatMessage {
            role: ChatRole::User,
            content: "hi".into(),
        });
        cm.clear();
        assert!(cm.messages().is_empty());
    }

    #[test]
    fn slide_keeps_last_n_turns() {
        let mut cm = ConversationManager::with_system_prompt("sys".into());
        for i in 0..4 {
            cm.push(ChatMessage {
                role: ChatRole::User,
                content: format!("q{i}"),
            });
            cm.push(ChatMessage {
                role: ChatRole::Assistant,
                content: format!("a{i}"),
            });
        }
        // 1 system + 8 messages = 9
        assert_eq!(cm.messages().len(), 9);

        cm.slide(2); // keep last 2 turns
        // 1 system + 4 messages = 5
        assert_eq!(cm.messages().len(), 5);
        assert_eq!(cm.messages()[0].role, ChatRole::System);
        assert_eq!(cm.messages()[1].content, "q2");
        assert_eq!(cm.messages()[4].content, "a3");
    }
}
