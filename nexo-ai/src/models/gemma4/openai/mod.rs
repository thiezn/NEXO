mod request_model;

use std::path::Path;

use crate::api::types::ToolCall;
use crate::models::gemma4::common::template::Gemma4Template;
use crate::models::support::prompting::ChatTemplate;
use crate::openai::model::{OpenAiFamilyAdapter, parse_wire_tool_calls};
use crate::openai::protocol::OpenAiResponseMessage;

pub use request_model::default_request_model_id;

#[derive(Debug, Clone, Default)]
pub struct Gemma4OpenAiFamily;

impl OpenAiFamilyAdapter for Gemma4OpenAiFamily {
    fn family(&self) -> &'static str {
        "gemma4"
    }

    fn resolve_request_model_id(
        &self,
        model_name: &str,
        model_dir: &Path,
        explicit: Option<&str>,
    ) -> String {
        explicit
            .map(str::to_string)
            .unwrap_or_else(|| default_request_model_id(model_name, model_dir))
    }

    fn parse_tool_response(
        &self,
        message: &OpenAiResponseMessage,
    ) -> (Vec<ToolCall>, Option<String>) {
        let template = Gemma4Template;
        let wire_tool_calls = parse_wire_tool_calls(&message.tool_calls);
        let raw_text = message.content.clone().unwrap_or_default();
        let (fallback_tool_calls, fallback_reasoning) = template.parse_tool_calls(&raw_text);
        let tool_calls = if wire_tool_calls.is_empty() {
            fallback_tool_calls
        } else {
            wire_tool_calls
        };
        let reasoning = message
            .content
            .clone()
            .filter(|text| !text.trim().is_empty())
            .or(fallback_reasoning);

        (tool_calls, reasoning)
    }
}
