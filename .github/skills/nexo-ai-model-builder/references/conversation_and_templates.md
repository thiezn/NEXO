# Conversation and Templates

## Current Paths

Shared prompting primitives live in:

- `nexo-ai/src/models/support/prompting.rs`
- `nexo-ai/src/models/support/conversation.rs`

Family-specific templates belong next to the family, usually in `nexo-ai/src/models/<family>/common/template.rs`.

Current reference implementation:

- `nexo-ai/src/models/gemma4/common/template.rs`

Do not move template logic into `models/support/`. `models/support/` is only for traits, shared helpers, and conversation scaffolding.

## `ChatTemplate`

`ChatTemplate` is the contract for Chat and Tool-capable families:

```rust
pub trait ChatTemplate: Send {
    fn format_prompt(&self, messages: &[ChatMessage], reasoning: &ReasoningMode) -> String;
    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[nexo_spec::tool::ToolSpec],
        reasoning: &ReasoningMode,
    ) -> String;
    fn parse_response(&self, raw: &str) -> String;
    fn parse_tool_calls(&self, raw: &str) -> (Vec<ToolCall>, Option<String>);
    fn end_of_turn_markers(&self) -> &[&str];
}
```

Use it to keep prompt formatting and raw-output parsing family-local. Local Candle backends and OpenAI-backed family adapters can share the same template behavior.

## `ReasoningMode`

`ReasoningMode` lives in `models/support/prompting.rs`.

- `Disabled`
- `Auto`
- `Effort(ReasoningEffort)`

Treat it as a capability surface, not a guarantee that every family uses every mode. Current Gemma4 template ignores the value, which is fine for a family that does not expose reasoning controls.

## `ConversationManager`

`ConversationManager` in `models/support/conversation.rs` owns message history and delegates formatting to a `ChatTemplate`.

Important methods:

- `push(msg)`
- `messages()`
- `clear()`
- `slide(keep_turns)`
- `summarize_prefix(summary)`
- `format(template)`
- `format_with_tools(template, tools)`
- `set_reasoning(mode)`

Use it from the REPL or other long-running chat surfaces. Do not re-implement rolling history inside individual model families.

## Family Guidance

- Keep templates in the family module, usually under `common/`.
- Put reusable family helpers next to the template instead of in `models/support/`.
- Provider-backed chat families can reuse local parsing. `nexo-ai/src/models/gemma4/openai/mod.rs` is the current example: it falls back to the Gemma4 template parser when wire tool calls are absent.
- `format_tools_xml()` and `format_tools_json()` in `models/support/prompting.rs` are optional helpers. They are not the architecture. Use them only if the family's prompt format actually matches.

## Minimum Template Tests

Add `#[cfg(test)]` coverage in the template file itself for:

- single user turn formatting
- system prompt handling
- multi-turn formatting
- tool declaration formatting
- tool response turns (`ChatRole::Tool`)
- tool-call parsing and fallback parsing
- end-of-turn stripping in `parse_response()`
