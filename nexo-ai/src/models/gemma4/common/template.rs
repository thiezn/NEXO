use crate::api::types::{ChatMessage, ChatRole, ToolCall};
use crate::models::support::prompting::{ChatTemplate, ReasoningMode};

pub struct Gemma4Template;

impl ChatTemplate for Gemma4Template {
    fn format_prompt(&self, messages: &[ChatMessage], _reasoning: &ReasoningMode) -> String {
        let mut prompt = String::new();

        for (i, msg) in messages.iter().enumerate() {
            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
                ChatRole::Tool => {
                    let needs_turn = i == 0
                        || !matches!(messages[i - 1].role, ChatRole::Assistant | ChatRole::Tool);
                    if needs_turn {
                        prompt.push_str("<|turn>model\n");
                    }
                    prompt.push_str("<|tool_response>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<tool_response|>");
                    if needs_turn {
                        let next_is_model = messages
                            .get(i + 1)
                            .is_some_and(|m| matches!(m.role, ChatRole::Assistant));
                        if !next_is_model {
                            prompt.push_str("<turn|>\n");
                        }
                    }
                    continue;
                }
            };

            let next_is_tool = messages
                .get(i + 1)
                .is_some_and(|m| matches!(m.role, ChatRole::Tool));

            prompt.push_str("<|turn>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            if !(msg.role == ChatRole::Assistant && next_is_tool) {
                prompt.push_str("<turn|>\n");
            }
        }

        prompt.push_str("<|turn>model\n");
        prompt
    }

    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[nexo_spec::tool::ToolSpec],
        reasoning: &ReasoningMode,
    ) -> String {
        let augmented = build_tool_messages(messages, tools);
        self.format_prompt(&augmented, reasoning)
    }

    fn parse_response(&self, raw: &str) -> String {
        let mut text = raw.to_string();

        while let Some(start) = text.find("<|channel>") {
            if let Some(end) = text[start..].find("<channel|>") {
                let end_abs = start + end + "<channel|>".len();
                text.replace_range(start..end_abs, "");
            } else {
                break;
            }
        }

        for marker in self.end_of_turn_markers() {
            if let Some(pos) = text.find(marker) {
                text.truncate(pos);
            }
        }

        text.trim().to_string()
    }

    fn parse_tool_calls(&self, raw: &str) -> (Vec<ToolCall>, Option<String>) {
        let mut tool_calls = Vec::new();
        let mut reasoning = None;

        let mut cleaned = raw.to_string();
        if let Some(start) = cleaned.find("<|channel>thought") {
            if let Some(end) = cleaned[start..].find("<channel|>") {
                let thought_content = &cleaned[start + "<|channel>thought".len()..start + end];
                let thought = thought_content.trim();
                if !thought.is_empty() {
                    reasoning = Some(thought.to_string());
                }
                let end_abs = start + end + "<channel|>".len();
                cleaned.replace_range(start..end_abs, "");
            }
        }

        let native_calls = parse_native_tool_calls(&cleaned);
        if !native_calls.is_empty() {
            return (native_calls, reasoning);
        }

        for marker in self.end_of_turn_markers() {
            if let Some(pos) = cleaned.find(marker) {
                cleaned.truncate(pos);
            }
        }
        let cleaned = cleaned.trim().to_string();

        let work = if let Some(start) = cleaned.find("```json") {
            let after = &cleaned[start + 7..];
            if let Some(end) = after.find("```") {
                let before = cleaned[..start].trim();
                if !before.is_empty() && reasoning.is_none() {
                    reasoning = Some(before.to_string());
                }
                after[..end].to_string()
            } else {
                after.to_string()
            }
        } else {
            cleaned
        };

        for candidate in extract_json_objects(&work) {
            if let Ok(tc) = serde_json::from_str::<ToolCall>(&candidate) {
                tool_calls.push(tc);
            }
        }

        if tool_calls.is_empty() && reasoning.is_none() {
            let fallback = work.trim();
            if !fallback.is_empty() {
                reasoning = Some(fallback.to_string());
            }
        }

        (tool_calls, reasoning)
    }

    fn end_of_turn_markers(&self) -> &[&str] {
        &["<turn|>", "<|tool_response>"]
    }
}

pub fn build_tool_messages(
    messages: &[ChatMessage],
    tools: &[nexo_spec::tool::ToolSpec],
) -> Vec<ChatMessage> {
    let mut tool_decls = String::new();
    for tool in tools {
        tool_decls.push_str(&format_tool_declaration(tool));
    }

    let system_text = format!(
        "You are a helpful assistant. When you need to use a tool, \
         respond with a tool_call block.{tool_decls}"
    );

    let mut augmented = Vec::with_capacity(messages.len() + 1);
    augmented.push(ChatMessage::new(ChatRole::System, system_text));
    augmented.extend_from_slice(messages);
    augmented
}

fn format_tool_declaration(tool: &nexo_spec::tool::ToolSpec) -> String {
    let mut decl = String::from("<|tool>declaration:");
    decl.push_str(&tool.name);
    decl.push('{');

    decl.push_str("description:");
    push_gemma4_string(&mut decl, &tool.description);

    if let Some(props) = tool.parameters.get("properties") {
        if let Some(obj) = props.as_object() {
            decl.push_str(",parameters:{");
            let mut first = true;
            for (name, schema) in obj {
                if !first {
                    decl.push(',');
                }
                first = false;
                decl.push_str(name);
                decl.push_str(":{");

                let mut inner_first = true;
                if let Some(ty) = schema.get("type").and_then(|v| v.as_str()) {
                    decl.push_str("type:");
                    push_gemma4_string(&mut decl, ty);
                    inner_first = false;
                }
                if let Some(desc) = schema.get("description").and_then(|v| v.as_str()) {
                    if !inner_first {
                        decl.push(',');
                    }
                    decl.push_str("description:");
                    push_gemma4_string(&mut decl, desc);
                }
                decl.push('}');
            }
            decl.push('}');
        }
    }

    decl.push('}');
    decl.push_str("<tool|>");
    decl
}

fn push_gemma4_string(out: &mut String, value: &str) {
    out.push_str("<|\"|>");
    out.push_str(value);
    out.push_str("<|\"|>");
}

pub(crate) fn parse_native_tool_calls(raw: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search = raw;

    while let Some(start) = search.find("<|tool_call>call:") {
        let after = &search[start + "<|tool_call>call:".len()..];
        if let Some(end) = after.find("<tool_call|>") {
            let content = &after[..end];
            if let Some(brace_start) = content.find('{') {
                let name = content[..brace_start].trim();
                let body = &content[brace_start + 1..];
                let body = body.strip_suffix('}').unwrap_or(body);
                let args = parse_gemma4_kv_pairs(body);
                calls.push(ToolCall {
                    name: name.to_string(),
                    arguments: serde_json::Value::Object(args),
                });
            }
            search = &after[end + "<tool_call|>".len()..];
        } else {
            break;
        }
    }
    calls
}

const STR_DELIM: &[u8] = b"<|\"|>";

fn bytes_match_at(bytes: &[u8], pos: usize, needle: &[u8]) -> bool {
    bytes
        .get(pos..pos + needle.len())
        .is_some_and(|s| s == needle)
}

fn parse_gemma4_kv_pairs(input: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    let input = input.trim();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        while pos < len && matches!(bytes[pos], b',' | b' ' | b'\n' | b'\r') {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        let key_start = pos;
        while pos < len && bytes[pos] != b':' {
            pos += 1;
        }
        if pos >= len {
            break;
        }
        let key = String::from_utf8_lossy(&bytes[key_start..pos])
            .trim()
            .to_string();
        pos += 1;

        if key.is_empty() {
            continue;
        }

        if bytes_match_at(bytes, pos, STR_DELIM) {
            pos += STR_DELIM.len();
            let str_start = pos;
            while pos < len && !bytes_match_at(bytes, pos, STR_DELIM) {
                pos += 1;
            }
            let value = String::from_utf8_lossy(&bytes[str_start..pos]).into_owned();
            if bytes_match_at(bytes, pos, STR_DELIM) {
                pos += STR_DELIM.len();
            }
            map.insert(key, serde_json::Value::String(value));
        } else if pos < len && bytes[pos] == b'{' {
            let mut depth = 0;
            let obj_start = pos;
            while pos < len {
                match bytes[pos] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            pos += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                pos += 1;
            }
            let inner =
                std::str::from_utf8(&bytes[obj_start + 1..pos.saturating_sub(1)]).unwrap_or("");
            let nested = parse_gemma4_kv_pairs(inner);
            map.insert(key, serde_json::Value::Object(nested));
        } else {
            let val_start = pos;
            while pos < len && !matches!(bytes[pos], b',' | b'}') {
                pos += 1;
            }
            let val_str = std::str::from_utf8(&bytes[val_start..pos])
                .unwrap_or("")
                .trim();
            let value = match val_str {
                "true" => serde_json::Value::Bool(true),
                "false" => serde_json::Value::Bool(false),
                "null" => serde_json::Value::Null,
                _ => {
                    if let Ok(n) = val_str.parse::<i64>() {
                        serde_json::Value::Number(n.into())
                    } else if let Ok(n) = val_str.parse::<f64>() {
                        serde_json::Number::from_f64(n)
                            .map(serde_json::Value::Number)
                            .unwrap_or_else(|| serde_json::Value::String(val_str.to_string()))
                    } else {
                        serde_json::Value::String(val_str.to_string())
                    }
                }
            };
            map.insert(key, value);
        }
    }

    map
}

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
                        i += 1;
                        while i < chars.len() && chars[i] != '"' {
                            if chars[i] == '\\' {
                                i += 1;
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
    fn build_tool_messages_inserts_system_instruction() {
        let augmented = build_tool_messages(
            &[ChatMessage::new(ChatRole::User, "Use a tool.")],
            &[nexo_spec::tool::ToolSpec {
                name: "get_weather".into(),
                description: "Get weather for a location".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    }
                }),
                ..Default::default()
            }],
        );

        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0].role, ChatRole::System);
        assert!(augmented[0].content.contains("tool_call block"));
        assert!(
            augmented[0]
                .content
                .contains("<|tool>declaration:get_weather{")
        );
    }

    #[test]
    fn format_simple_prompt() {
        let msgs = vec![ChatMessage::new(ChatRole::User, "Hello!")];

        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);

        assert!(result.contains("<|turn>user\nHello!<turn|>"));
        assert!(result.ends_with("<|turn>model\n"));
    }

    #[test]
    fn format_with_system() {
        let msgs = vec![
            ChatMessage::new(ChatRole::System, "You are helpful."),
            ChatMessage::new(ChatRole::User, "Hi"),
        ];

        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);

        assert!(result.contains("<|turn>system\nYou are helpful.<turn|>"));
        assert!(result.contains("<|turn>user\nHi<turn|>"));
        assert!(result.ends_with("<|turn>model\n"));
    }

    #[test]
    fn format_multi_turn() {
        let msgs = vec![
            ChatMessage::new(ChatRole::User, "Hi"),
            ChatMessage::new(ChatRole::Assistant, "Hello!"),
            ChatMessage::new(ChatRole::User, "How are you?"),
        ];

        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);

        assert!(result.contains("<|turn>user\nHi<turn|>"));
        assert!(result.contains("<|turn>model\nHello!<turn|>"));
        assert!(result.contains("<|turn>user\nHow are you?<turn|>"));
        assert!(result.ends_with("<|turn>model\n"));
    }

    #[test]
    fn parse_response_strips_turn_marker() {
        assert_eq!(
            template().parse_response("Hello there!<turn|>"),
            "Hello there!"
        );
    }

    #[test]
    fn parse_response_strips_tool_response_marker() {
        assert_eq!(
            template().parse_response("I'll check that.<|tool_response>"),
            "I'll check that."
        );
    }

    #[test]
    fn parse_response_strips_thinking_channel() {
        let raw = "<|channel>thought\nLet me think...\n<channel|>The answer is 4.";
        assert_eq!(template().parse_response(raw), "The answer is 4.");
    }

    #[test]
    fn parse_native_tool_call() {
        let raw = r#"<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|><|tool_response>"#;
        let (calls, _) = template().parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "London");
    }

    #[test]
    fn parse_native_tool_call_with_number() {
        let raw = r#"<|tool_call>call:set_temp{value:22,unit:<|"|>celsius<|"|>}<tool_call|>"#;
        let (calls, _) = template().parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "set_temp");
        assert_eq!(calls[0].arguments["value"], 22);
        assert_eq!(calls[0].arguments["unit"], "celsius");
    }

    #[test]
    fn parse_native_tool_call_utf8_value() {
        let raw =
            "<|tool_call>call:search{query:<|\"|>caf\u{00e9} in Z\u{00fc}rich<|\"|>}<tool_call|>";
        let (calls, _) = template().parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["query"], "caf\u{00e9} in Z\u{00fc}rich");
    }

    #[test]
    fn parse_tool_calls_json_fallback() {
        let raw = "I'll look that up.\n```json\n{\"name\": \"search\", \"arguments\": {\"query\": \"weather\"}}\n```";
        let (calls, reasoning) = template().parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["query"], "weather");
        assert_eq!(reasoning.unwrap(), "I'll look that up.");
    }

    #[test]
    fn parse_tool_calls_returns_plain_text_when_no_calls_present() {
        let raw = "The answer is 4.<turn|>";
        let (calls, reasoning) = template().parse_tool_calls(raw);
        assert!(calls.is_empty());
        assert_eq!(reasoning.unwrap(), "The answer is 4.");
    }

    #[test]
    fn extract_json_objects_multiple() {
        let objects = extract_json_objects(r#"{"a":1} some text {"b":2}"#);

        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn extract_json_objects_nested() {
        let objects = extract_json_objects(r#"{"name":"test","arguments":{"key":"value"}}"#);
        let tool_call: ToolCall = serde_json::from_str(&objects[0]).unwrap();

        assert_eq!(objects.len(), 1);
        assert_eq!(tool_call.name, "test");
    }

    #[test]
    fn tool_response_merges_into_model_turn() {
        let msgs = vec![
            ChatMessage::new(ChatRole::User, "What's the weather?"),
            ChatMessage::new(ChatRole::Assistant, "Let me check."),
            ChatMessage::new(ChatRole::Tool, "temperature: 15C"),
        ];

        let result = template().format_prompt(&msgs, &ReasoningMode::Disabled);

        assert!(result.contains(
            "<|turn>model\nLet me check.<|tool_response>temperature: 15C<tool_response|>"
        ));
        assert!(result.ends_with("<|turn>model\n"));
    }
}
