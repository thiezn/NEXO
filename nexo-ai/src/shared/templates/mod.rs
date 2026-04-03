pub mod conversation;
pub mod kv_cache;

use crate::shared::types::{ChatMessage, ToolCall};

/// Controls how a model handles reasoning/thinking.
/// Different model families map their capabilities onto these variants.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReasoningMode {
    /// No reasoning output (e.g., Qwen3 enable_thinking=false).
    Disabled,
    /// Model decides whether to reason (e.g., Qwen3 enable_thinking=true).
    ///
    #[default]
    Auto,
    /// Explicit effort level for models that support it.
    Effort(ReasoningEffort),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

/// Defines how a model family formats conversations into prompt strings.
pub trait ChatTemplate: Send {
    /// Format conversation messages into a model-ready prompt string.
    fn format_prompt(&self, messages: &[ChatMessage], reasoning: &ReasoningMode) -> String;

    /// Format conversation with tool definitions embedded.
    fn format_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[nexo_spec::tool::ToolSpec],
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

// ── Shared formatting helpers ──────────────────────────────────────────────

/// Format tool specs as `<tools>` XML with JSON objects (Qwen3 official style).
pub fn format_tools_xml(tools: &[nexo_spec::tool::ToolSpec]) -> String {
    let mut out = String::from("<tools>");
    for tool in tools {
        out.push('\n');
        out.push_str(&serde_json::to_string(tool).unwrap_or_default());
    }
    out.push_str("\n</tools>");
    out
}

/// Format tool specs as a pretty-printed JSON array (Gemma3 style).
pub fn format_tools_json(tools: &[nexo_spec::tool::ToolSpec]) -> String {
    serde_json::to_string_pretty(tools).unwrap_or_default()
}
