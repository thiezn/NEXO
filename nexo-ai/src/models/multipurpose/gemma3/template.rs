use crate::shared::types::{ChatMessage, ChatRole, ToolCall};

pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut system_text: Option<&str> = None;

    // Collect system message to prepend to first user turn.
    for msg in messages {
        if msg.role == ChatRole::System {
            system_text = Some(&msg.content);
            break;
        }
    }

    for msg in messages {
        match msg.role {
            ChatRole::System => {}
            ChatRole::User => {
                prompt.push_str("<start_of_turn>user\n");
                if let Some(sys) = system_text.take() {
                    prompt.push_str(sys);
                    prompt.push('\n');
                }
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
        }
    }

    // Final model turn for generation.
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

pub fn format_tool_prompt(
    messages: &[ChatMessage],
    tools: &[nexo_tool_spec::tool::ToolSpec],
) -> String {
    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();

    let system_instruction = format!(
        "You have access to the following tools:\n\n{tools_json}\n\n\
         When you need to call a tool, respond with a JSON object like:\n\
         {{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n\n\
         If you don't need to call a tool, respond normally."
    );

    let mut augmented = vec![ChatMessage {
        role: ChatRole::System,
        content: system_instruction,
    }];

    for msg in messages {
        if msg.role != ChatRole::System {
            augmented.push(msg.clone());
        }
    }

    format_chat_prompt(&augmented)
}

pub fn parse_tool_response(text: &str) -> (Vec<ToolCall>, Option<String>) {
    let trimmed = text.trim();

    // Try parsing the entire response as a tool call.
    if let Ok(call) = serde_json::from_str::<ToolCall>(trimmed) {
        return (vec![call], None);
    }

    // Try parsing as an array of tool calls.
    if let Ok(calls) = serde_json::from_str::<Vec<ToolCall>>(trimmed) {
        if !calls.is_empty() {
            return (calls, None);
        }
    }

    // Try to find JSON object(s) embedded in text.
    let mut calls = Vec::new();
    let mut reasoning = String::new();
    let mut search_from = 0;

    while let Some(start) = trimmed[search_from..].find('{') {
        let abs_start = search_from + start;
        // Collect text before JSON as reasoning.
        if calls.is_empty() {
            let before = trimmed[search_from..abs_start].trim();
            if !before.is_empty() {
                reasoning.push_str(before);
            }
        }

        // Find matching closing brace (handles nested braces).
        let mut depth = 0;
        let mut end = None;
        for (i, ch) in trimmed[abs_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(abs_start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }

        let Some(abs_end) = end else {
            break;
        };

        if let Ok(call) = serde_json::from_str::<ToolCall>(&trimmed[abs_start..abs_end]) {
            calls.push(call);
        }
        search_from = abs_end;
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

    #[test]
    fn single_user_message() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".into(),
        }];
        let prompt = format_chat_prompt(&msgs);
        assert_eq!(
            prompt,
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn system_prepended_to_first_user() {
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
        let prompt = format_chat_prompt(&msgs);
        assert!(prompt.contains("<start_of_turn>user\nBe helpful.\nHi<end_of_turn>"));
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
        let prompt = format_chat_prompt(&msgs);
        assert!(prompt.contains("<start_of_turn>model\nreply<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn parse_single_tool_call() {
        let text = r#"{"name": "get_weather", "arguments": {"city": "Amsterdam"}}"#;
        let (calls, reasoning) = parse_tool_response(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(reasoning.is_none());
    }

    #[test]
    fn parse_tool_call_with_reasoning() {
        let text = "I need to check the weather.\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}";
        let (calls, reasoning) = parse_tool_response(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(reasoning.unwrap().contains("check the weather"));
    }

    #[test]
    fn parse_no_tool_calls() {
        let text = "I don't need any tools for this.";
        let (calls, _) = parse_tool_response(text);
        assert!(calls.is_empty());
    }
}
