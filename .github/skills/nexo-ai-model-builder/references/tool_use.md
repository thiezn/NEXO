# Tool Use

## Overview

There is no separate tool-model templating trait. Tool use is driven by the same family-local prompt parser used for chat:

- `ChatTemplate::format_with_tools()`
- `ChatTemplate::parse_tool_calls()`

Tool definitions still use `nexo_spec::tool::ToolSpec`.

## Current Reference Points

- `nexo-ai/src/models/support/prompting.rs`
- `nexo-ai/src/models/gemma4/common/template.rs`
- `nexo-ai/src/models/gemma4/openai/mod.rs`

Gemma4 is the main reference implementation for local tool prompting and parsing. The OpenAI-backed Gemma4 adapter shows how to combine provider-native tool-call data with family-local fallback parsing.

## Local Family Pattern

For local Candle families:

1. Format tool declarations inside `format_with_tools()`.
2. Handle `ChatRole::Tool` explicitly in `format_prompt()`.
3. Parse structured tool calls in `parse_tool_calls()`.
4. Return `(Vec<ToolCall>, Option<String>)`, preserving reasoning text when the family emits it.

Do not assume XML or JSON helper functions are mandatory. `format_tools_xml()` and `format_tools_json()` are convenience helpers only.

## OpenAI-Backed Chat Families

For provider-backed chat models using `nexo-ai/src/openai/model.rs`:

- Provider-native tool calls are normalized by `parse_wire_tool_calls()`.
- Families can override `OpenAiFamilyAdapter::parse_tool_response()` when provider output is incomplete or family-specific parsing is better.
- `nexo-ai/src/models/gemma4/openai/mod.rs` is the current reference. It prefers wire tool calls when present and falls back to the Gemma4 template parser when they are not.

## Parsing Guidance

- Support the family's native tool-call syntax first.
- Add a JSON-object fallback when the model may emit raw JSON instead of the ideal wrapper.
- Strip end-of-turn markers before parsing fallback content.
- Keep reasoning text separate from the structured `ToolCall` list when possible.

## Testing

Use `tool_test!` in `nexo-ai/tests/model_inference.rs` for local chat/tool families.

Also add unit tests in the family template file for:

- native tool-call syntax
- JSON fallback parsing
- reasoning + tool-call coexistence
- `ChatRole::Tool` rendering back into the prompt
