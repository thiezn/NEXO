# Tool Use Implementation

## Overview

Tool use is handled within the `ChatTemplate` trait -- there is no separate tool-specific trait. Two methods are responsible:

- `format_with_tools()` -- embeds tool definitions into the prompt
- `parse_tool_calls()` -- extracts structured tool calls from model output

Tool definitions use `nexo_tool_spec::tool::ToolSpec` from the `nexo-tool-spec` crate. No changes to that crate are needed when adding a new model.

## Implementing `format_with_tools`

The general pattern:

1. Build a tool instruction string describing the available tools
2. Merge it with any existing system message content
3. Construct an augmented message list with the combined system message
4. Delegate to `self.format_prompt()`

### Shared Formatting Helpers

`shared/templates/mod.rs` provides two helpers:

```rust
/// XML format: <tools>\n{json}\n{json}\n</tools>  (Qwen3 style)
pub fn format_tools_xml(tools: &[ToolSpec]) -> String;

/// Pretty-printed JSON array  (Gemma3 style)
pub fn format_tools_json(tools: &[ToolSpec]) -> String;
```

Choose the format that matches the model's official template. Most models use one or the other.

### Example: Qwen3 Pattern

```rust
fn format_with_tools(&self, messages: &[ChatMessage], tools: &[ToolSpec], reasoning: &ReasoningMode) -> String {
    let tools_section = format_tools_xml(tools);
    let tool_instruction = format!("# Tools\n\nYou may call one or more functions...\n{tools_section}\n...");

    // Single-pass: separate system messages from non-system, merge system with tool instruction
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

    augmented.insert(0, ChatMessage { role: ChatRole::System, content: full_system });
    self.format_prompt(&augmented, reasoning)
}
```

## Implementing `parse_tool_calls`

Returns `(Vec<ToolCall>, Option<String>)` -- the extracted calls and any reasoning text that preceded them.

### Common Extraction Strategies

**Tag-based (Qwen3):** Look for `<tool_call>...</tool_call>` XML tags, parse JSON content inside each.

**JSON detection (Gemma3):** Try full-response JSON parse first, then fall back to brace-matching to find embedded JSON objects.

Both implementations include a raw JSON fallback for when the model produces a bare JSON object without wrapper tags.

### ToolCall Structure

```rust
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
```

The `arguments` field is a `serde_json::Value`, not a string. When serializing tool calls into prompts (for multi-turn tool use), check if arguments is already a Value before calling `serde_json::to_string`.

## Tool Role in Conversations

`ChatRole::Tool` represents tool response messages in the conversation. How these are formatted depends on the model family:

- **Qwen3**: Tool messages are wrapped in `<tool_response>` tags inside a `user` turn. Consecutive tool messages are grouped into a single user turn.
- **Gemma3**: Tool messages are treated as regular user turns.

When implementing `format_prompt`, handle the `ChatRole::Tool` variant explicitly.

## Testing Tool Use

Use the `tool_test!` macro in `tests/model_inference.rs`:

```rust
tool_test!(test_my_model_tool, "my-model-name", nexo_ai::models::multipurpose::my_family::MyModel);
```

The macro provides a standard `get_weather` tool and validates the model produces at least one token. Add model-specific tests in the template's `#[cfg(test)]` module for parsing edge cases.
