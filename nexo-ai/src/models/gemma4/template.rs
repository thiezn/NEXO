use crate::shared::templates::{format_tools_json, ChatTemplate, ReasoningMode};
use crate::shared::types::{ChatMessage, ChatRole, ToolCall};

/// Gemma 4 chat template.
///
/// Format:
/// ```text
/// <start_of_turn>user
/// {content}<end_of_turn>
/// <start_of_turn>model
/// {content}<end_of_turn>
/// ```
pub struct Gemma4Template;

impl ChatTemplate for Gemma4Template {
    fn format_prompt(&self, messages: &[ChatMessage], _reasoning: &ReasoningMode) -> String {
        let mut prompt = String::new();

        for msg in messages {
            let role = match msg.role {
                ChatRole::System => "user",
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
                ChatRole::Tool => "user",
            };
            prompt.push_str("<start_of_turn>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<end_of_turn>\n");
        }

        // Prompt the model to start generating
        prompt.push_str("<start_of_turn>model\n");
        prompt
    }

    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[nexo_spec::tool::ToolSpec],
        reasoning: &ReasoningMode,
    ) -> String {
        // Inject tool definitions into the system prompt
        let tool_text = format!(
            "You have access to the following tools:\n\n{}\n\n\
             To use a tool, respond with a JSON object in the format:\n\
             ```json\n\
             {{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n\
             ```\n\
             You may call multiple tools by returning multiple JSON objects.",
            format_tools_json(tools)
        );

        let mut augmented = Vec::with_capacity(messages.len() + 1);
        augmented.push(ChatMessage {
            role: ChatRole::System,
            content: tool_text,
        });
        augmented.extend_from_slice(messages);

        self.format_prompt(&augmented, reasoning)
    }

    fn parse_response(&self, raw: &str) -> String {
        let mut text = raw.to_string();

        // Strip end-of-turn markers
        for marker in self.end_of_turn_markers() {
            if let Some(pos) = text.find(marker) {
                text.truncate(pos);
            }
        }

        text.trim().to_string()
    }

    fn parse_tool_calls(&self, raw: &str) -> (Vec<ToolCall>, Option<String>) {
        let cleaned = self.parse_response(raw);
        let mut tool_calls = Vec::new();
        let mut reasoning = None;

        // Try to extract JSON objects from the response
        // Look for ```json blocks first
        let work = if let Some(start) = cleaned.find("```json") {
            let after = &cleaned[start + 7..];
            if let Some(end) = after.find("```") {
                // Content before the code block is reasoning
                let before = cleaned[..start].trim();
                if !before.is_empty() {
                    reasoning = Some(before.to_string());
                }
                after[..end].to_string()
            } else {
                after.to_string()
            }
        } else {
            cleaned
        };

        // Parse individual JSON objects
        for candidate in extract_json_objects(&work) {
            if let Ok(tc) = serde_json::from_str::<ToolCall>(&candidate) {
                tool_calls.push(tc);
            }
        }

        (tool_calls, reasoning)
    }

    fn end_of_turn_markers(&self) -> &[&str] {
        &["<end_of_turn>", "<eos>"]
    }
}

/// Extract top-level JSON objects from a string by brace matching.
fn extract_json_objects(input: &str) -> Vec<String> {
    let mut results = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            let start = i;
            let mut depth = 1;
            i += 1;
            while i < chars.len() && depth > 0 {
                match chars[i] {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    '"' => {
                        // Skip string contents
                        i += 1;
                        while i < chars.len() && chars[i] != '"' {
                            if chars[i] == '\\' {
                                i += 1; // skip escaped char
                            }
                            i += 1;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            if depth == 0 {
                results.push(chars[start..i].iter().collect());
            }
        } else {
            i += 1;
        }
    }

    results
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn template() -> Gemma4Template {
        Gemma4Template
    }

    #[test]
    fn format_simple_prompt() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello!".into(),
        }];
        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);
        assert!(result.contains("<start_of_turn>user\nHello!<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn format_multi_turn() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Hi".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "Hello!".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "How are you?".into(),
            },
        ];
        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);
        assert!(result.contains("<start_of_turn>user\nHi<end_of_turn>"));
        assert!(result.contains("<start_of_turn>model\nHello!<end_of_turn>"));
        assert!(result.contains("<start_of_turn>user\nHow are you?<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn parse_response_strips_markers() {
        let raw = "Hello there!<end_of_turn>";
        assert_eq!(template().parse_response(raw), "Hello there!");
    }

    #[test]
    fn parse_response_strips_eos() {
        let raw = "Hello there!<eos>";
        assert_eq!(template().parse_response(raw), "Hello there!");
    }

    #[test]
    fn parse_tool_calls_json_block() {
        let raw = "I'll look that up.\n```json\n{\"name\": \"search\", \"arguments\": {\"query\": \"weather\"}}\n```";
        let (calls, reasoning) = template().parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["query"], "weather");
        assert_eq!(reasoning.unwrap(), "I'll look that up.");
    }

    #[test]
    fn extract_json_objects_multiple() {
        let input = r#"{"a":1} some text {"b":2}"#;
        let objects = extract_json_objects(input);
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn extract_json_objects_nested() {
        let input = r#"{"name":"test","arguments":{"key":"value"}}"#;
        let objects = extract_json_objects(input);
        assert_eq!(objects.len(), 1);
        let tc: ToolCall = serde_json::from_str(&objects[0]).unwrap();
        assert_eq!(tc.name, "test");
    }

    #[test]
    fn end_of_turn_markers() {
        let t = template();
        let markers = t.end_of_turn_markers();
        assert!(markers.contains(&"<end_of_turn>"));
        assert!(markers.contains(&"<eos>"));
    }
}
