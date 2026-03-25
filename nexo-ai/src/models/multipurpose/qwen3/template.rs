use crate::shared::templates::{format_tools_xml, ChatTemplate, ReasoningMode};
use crate::shared::types::{ChatMessage, ChatRole, ToolCall};

pub struct Qwen3Template;

impl ChatTemplate for Qwen3Template {
    fn format_prompt(&self, messages: &[ChatMessage], reasoning: &ReasoningMode) -> String {
        let mut prompt = String::new();

        // Find the index of the last real user query (not a Tool message starting with
        // `<tool_response>`). We traverse in reverse to locate it.
        let last_user_idx = messages
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.role == ChatRole::User && !m.content.starts_with("<tool_response>"))
            .map(|(i, _)| i);

        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::System => {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                ChatRole::User => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                ChatRole::Assistant => {
                    let content = if let Some(user_idx) = last_user_idx {
                        if i < user_idx {
                            // Earlier assistant message: strip thinking blocks.
                            strip_thinking(&msg.content)
                        } else {
                            // After last user query: preserve thinking blocks.
                            msg.content.clone()
                        }
                    } else {
                        msg.content.clone()
                    };
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&content);
                    prompt.push_str("<|im_end|>\n");
                }
                ChatRole::Tool => {
                    // Group consecutive Tool messages into a single user turn with
                    // <tool_response> wrappers.
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str("<tool_response>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n</tool_response>");

                    // Consume any subsequent consecutive Tool messages.
                    while i + 1 < messages.len() && messages[i + 1].role == ChatRole::Tool {
                        i += 1;
                        prompt.push_str("\n<tool_response>\n");
                        prompt.push_str(&messages[i].content);
                        prompt.push_str("\n</tool_response>");
                    }

                    prompt.push_str("<|im_end|>\n");
                }
            }
            i += 1;
        }

        // Final assistant turn for generation.
        prompt.push_str("<|im_start|>assistant\n");

        // When reasoning is disabled, inject empty think tags so the model skips thinking.
        if *reasoning == ReasoningMode::Disabled {
            prompt.push_str("<think>\n\n</think>\n\n");
        }

        prompt
    }

    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[nexo_tool_spec::tool::ToolSpec],
        reasoning: &ReasoningMode,
    ) -> String {
        let tools_section = format_tools_xml(tools);

        let tool_instruction = format!(
            "\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_section}

For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags:
<tool_call>
{{\"name\": <function-name>, \"arguments\": <args-json-object>}}
</tool_call>"
        );

        let mut system_parts = Vec::new();
        let mut augmented = Vec::with_capacity(messages.len());

        for msg in messages {
            if msg.role == ChatRole::System {
                system_parts.push(msg.content.as_str());
            } else {
                augmented.push(msg.clone());
            }
        }

        let full_system = if system_parts.is_empty() {
            tool_instruction
        } else {
            format!("{}\n\n{tool_instruction}", system_parts.join("\n"))
        };

        augmented.insert(0, ChatMessage {
            role: ChatRole::System,
            content: full_system,
        });

        self.format_prompt(&augmented, reasoning)
    }

    fn parse_response(&self, raw: &str) -> String {
        strip_thinking(raw)
    }

    fn parse_tool_calls(&self, raw: &str) -> (Vec<ToolCall>, Option<String>) {
        parse_tool_response(raw)
    }

    fn end_of_turn_markers(&self) -> &[&str] {
        &["<|im_end|>"]
    }
}

/// Strip `<think>...</think>` blocks from model output.
pub fn strip_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> -- discard the rest
            return result.trim().to_string();
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

pub fn parse_tool_response(text: &str) -> (Vec<ToolCall>, Option<String>) {
    let cleaned = strip_thinking(text);
    let trimmed = cleaned.trim();

    let mut calls = Vec::new();
    let mut reasoning = String::new();
    let mut search_from = 0;

    while let Some(tag_start) = trimmed[search_from..].find("<tool_call>") {
        let abs_tag = search_from + tag_start;

        // Text before the first tool_call is reasoning.
        if calls.is_empty() {
            let before = trimmed[search_from..abs_tag].trim();
            if !before.is_empty() {
                reasoning.push_str(before);
            }
        }

        let content_start = abs_tag + "<tool_call>".len();
        if let Some(tag_end) = trimmed[content_start..].find("</tool_call>") {
            let json_str = trimmed[content_start..content_start + tag_end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                calls.push(call);
            }
            search_from = content_start + tag_end + "</tool_call>".len();
        } else {
            break;
        }
    }

    // If no <tool_call> tags found, try raw JSON (fallback).
    if calls.is_empty() {
        if let Ok(call) = serde_json::from_str::<ToolCall>(trimmed) {
            return (vec![call], None);
        }
    }

    let reasoning = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };

    (calls, reasoning)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn template() -> Qwen3Template {
        Qwen3Template
    }

    #[test]
    fn single_user_message() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".into(),
        }];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn system_message_separate_turn() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "Be helpful.".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hi".into(),
            },
        ];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        assert!(prompt.starts_with("<|im_start|>system\nBe helpful.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
    }

    #[test]
    fn multi_turn_conversation() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "first".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "reply".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "second".into(),
            },
        ];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        assert!(prompt.contains("<|im_start|>assistant\nreply<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn strip_thinking_removes_block() {
        let text = "<think>\nI need to think.\n</think>\n\nHello!";
        assert_eq!(strip_thinking(text), "Hello!");
    }

    #[test]
    fn strip_thinking_no_block() {
        let text = "Just a normal response.";
        assert_eq!(strip_thinking(text), "Just a normal response.");
    }

    #[test]
    fn strip_thinking_unclosed() {
        let text = "<think>thinking forever...";
        assert_eq!(strip_thinking(text), "");
    }

    #[test]
    fn parse_tool_call_in_tags() {
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Amsterdam"}}
</tool_call>"#;
        let (calls, reasoning) = parse_tool_response(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(reasoning.is_none());
    }

    #[test]
    fn parse_tool_call_with_thinking_and_reasoning() {
        let text = r#"<think>
Let me check the weather.
</think>

I'll look that up for you.
<tool_call>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool_call>"#;
        let (calls, reasoning) = parse_tool_response(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(reasoning.unwrap().contains("look that up"));
    }

    #[test]
    fn parse_no_tool_calls() {
        let text = "I don't need any tools for this.";
        let (calls, _) = parse_tool_response(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_raw_json_fallback() {
        let text = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let (calls, _) = parse_tool_response(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn reasoning_disabled_adds_empty_think_tags() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".into(),
        }];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Disabled);
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn reasoning_auto_does_not_add_think_tags() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".into(),
        }];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!prompt.contains("<think>"));
    }

    #[test]
    fn tool_role_messages_wrapped_in_tool_response() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "What is the weather?".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "<think>checking</think>\nLet me check.".into(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: r#"{"temperature": 20}"#.into(),
            },
        ];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        assert!(prompt.contains("<|im_start|>user\n<tool_response>\n{\"temperature\": 20}\n</tool_response><|im_end|>"));
    }

    #[test]
    fn consecutive_tool_messages_grouped_in_single_user_turn() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Do two things".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "calling tools".into(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: "result1".into(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: "result2".into(),
            },
        ];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        // Both tool responses should be in a single user turn.
        let user_turns: Vec<_> = prompt.matches("<|im_start|>user\n").collect();
        // One real user message + one grouped tool response turn = 2 user turns.
        assert_eq!(user_turns.len(), 2);
        assert!(prompt.contains("<tool_response>\nresult1\n</tool_response>\n<tool_response>\nresult2\n</tool_response>"));
    }

    #[test]
    fn earlier_assistant_thinking_stripped() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "first".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "<think>old reasoning</think>\nold reply".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "second".into(),
            },
        ];
        let prompt = template().format_prompt(&msgs, &ReasoningMode::Auto);
        // The earlier assistant message should have thinking stripped.
        assert!(prompt.contains("<|im_start|>assistant\nold reply<|im_end|>"));
        assert!(!prompt.contains("old reasoning"));
    }

    #[test]
    fn end_of_turn_markers_correct() {
        assert_eq!(template().end_of_turn_markers(), &["<|im_end|>"]);
    }
}
