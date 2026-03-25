# Conversation Management and Chat Templates

## Architecture Overview

```
shared/templates/
  mod.rs           -- ChatTemplate trait, ReasoningMode enum, shared formatting helpers
  conversation.rs  -- ConversationManager (history, formatting, context windowing)
  kv_cache.rs      -- KvCacheState trait (prefix caching abstraction)

models/multipurpose/qwen3/template.rs  -- Qwen3Template implements ChatTemplate
models/multipurpose/gemma3/template.rs -- Gemma3Template implements ChatTemplate
```

Model-specific template implementations live in their model folders, not in `shared/templates/`.

## ChatTemplate Trait

Every model that supports Chat or Tool categories must implement `ChatTemplate` (in `shared/templates/mod.rs`):

```rust
pub trait ChatTemplate: Send {
    /// Format conversation messages into a model-ready prompt string.
    fn format_prompt(&self, messages: &[ChatMessage], reasoning: &ReasoningMode) -> String;

    /// Format conversation with tool definitions embedded.
    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        reasoning: &ReasoningMode,
    ) -> String;

    /// Clean raw model output (strip thinking blocks, control tokens, etc.).
    fn parse_response(&self, raw: &str) -> String;

    /// Extract tool calls from raw model output.
    /// Returns (tool_calls, optional_reasoning).
    fn parse_tool_calls(&self, raw: &str) -> (Vec<ToolCall>, Option<String>);

    /// End-of-turn marker strings for this template family.
    fn end_of_turn_markers(&self) -> &[&str];
}
```

### Implementation Checklist

1. **`format_prompt`** -- Convert `&[ChatMessage]` into the model's native prompt format. Handle all `ChatRole` variants: `System`, `User`, `Assistant`, `Tool`.
2. **`format_with_tools`** -- Inject tool definitions into the system message, then delegate to `format_prompt`. Use the shared helpers: `format_tools_xml()` for XML-style (Qwen3) or `format_tools_json()` for JSON-style (Gemma3).
3. **`parse_response`** -- Strip thinking blocks, control tokens, or other non-content output. For models without thinking support, return the input unchanged.
4. **`parse_tool_calls`** -- Extract structured `ToolCall` objects from raw output. Return any reasoning text that preceded the tool calls.
5. **`end_of_turn_markers`** -- Return the strings that signal end of the model's turn (used for stopping generation).

### Reference Implementations

- **Qwen3** (`models/multipurpose/qwen3/template.rs`): `<|im_start|>/<|im_end|>` format, XML tool defs, `<tool_call>` tags, `<tool_response>` in user turns, rolling context management for `<think>` blocks.
- **Gemma3** (`models/multipurpose/gemma3/template.rs`): `<start_of_turn>user/model` format, system prepended to first user turn, JSON tool defs, brace-matching JSON extraction for tool calls.

## ReasoningMode

Controls how a model handles internal reasoning/thinking. Different model families map their capabilities onto these variants:

```rust
pub enum ReasoningMode {
    Disabled,                    // No reasoning output
    Auto,                        // Model decides whether to reason (default)
    Effort(ReasoningEffort),     // Explicit effort level
}

pub enum ReasoningEffort { Low, Medium, High, Max }
```

**Model family mapping:**

| Family | Disabled | Auto | Effort |
|--------|----------|------|--------|
| Qwen3 / DeepSeek | Prefill empty `<think>` tags | Let model decide | Same as Auto |
| Gemma3 | Ignored (no thinking support) | Ignored | Ignored |
| OpenAI (future) | `reasoning_effort: low` | Default | Maps to `reasoning_effort` |
| Claude (future) | Low effort | Default | Maps to effort level |

When implementing `format_prompt`, check `reasoning` and act accordingly. Models without thinking support should ignore it entirely (see Gemma3 using `_reasoning`).

## ConversationManager

`ConversationManager` (`shared/templates/conversation.rs`) manages multi-turn conversation history in the REPL. It is **model-agnostic** -- it stores messages and delegates formatting to a `ChatTemplate`.

### How it's used in the REPL

The REPL creates a `ConversationManager` before the main loop. On each chat message:

1. User input is pushed as `ChatRole::User`
2. `conversation.messages()` provides the full history to `ChatRequest`
3. Model response is pushed as `ChatRole::Assistant`
4. `/clear` resets the conversation (keeping system prompt if set)

### Context Management Methods

| Method | Purpose |
|--------|---------|
| `push(msg)` | Append a message |
| `clear()` | Reset history, keep system prompt |
| `slide(keep_turns)` | Keep system prompt + last N turn pairs |
| `summarize_prefix(summary)` | Replace older messages with a summary |
| `format(template)` | Format all messages using a `ChatTemplate` |
| `format_with_tools(template, tools)` | Format with tool definitions |

### Future: Prefix Caching

The `KvCacheState` trait (`shared/templates/kv_cache.rs`) abstracts KV cache operations for conversation-aware cache reuse:

```rust
pub trait KvCacheState {
    fn cache_token_count(&self) -> usize;
    fn clear_cache(&mut self);
    fn truncate_to(&mut self, len: usize);
}
```

Phase 1 (current): trait is defined, models implement stubs. Phase 2: real prefix caching with token-level truncation, allowing the KV cache to be reused across turns instead of re-encoding the entire conversation history each time.
