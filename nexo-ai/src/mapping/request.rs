use std::collections::HashMap;

use either::Either;
use indexmap::IndexMap;
use mistralrs_core::Function as MistralFunction;
use mistralrs_core::{
    Constraint, DetokenizationRequest as MistralDetokenizationRequest, MessageContent,
    NormalRequest, ReasoningEffort as MistralReasoningEffort, RequestMessage, Response,
    SamplingParams, StopTokens, TokenizationRequest as MistralTokenizationRequest, Tool,
    ToolChoice as MistralToolChoice, ToolType,
};
use nexo_core::inference::request::{DetokenizationRequest, GenerateRequest, TokenizationRequest};
use nexo_core::{
    ContentPart, Conversation, ConversationMessage, MessageRole, ModelDescriptor, OutputConstraint,
    RoleStrategy, SamplingConfig, SpecialTokenPolicy, TextPart, ToolCall, ToolChoice,
    ToolDefinition, ToolResultContent,
};
use tokio::sync::mpsc::Sender;

use crate::{Error, Result};

/// Builds a `mistralrs-core` normal request for conversational generation.
pub(crate) fn map_generate_request(
    request: &GenerateRequest,
    descriptor: &ModelDescriptor,
    response: Sender<Response>,
    request_ordinal: usize,
) -> Result<NormalRequest> {
    let tools = map_tool_definitions(&request.tools)?;

    Ok(NormalRequest {
        messages: RequestMessage::Chat {
            messages: map_conversation(&request.conversation, descriptor.role_strategy)?,
            enable_thinking: Some(thinking_enabled(request.reasoning.thinking)),
            reasoning_effort: map_reasoning_effort(request.reasoning.effort),
        },
        sampling_params: map_sampling(&request.sampling),
        response,
        return_logprobs: false,
        is_streaming: matches!(request.streaming, nexo_core::StreamingMode::Streaming),
        id: request_ordinal,
        constraint: map_constraint(&request.output_constraint),
        suffix: None,
        tools: tools.clone(),
        tool_choice: map_tool_choice(&request.tool_choice, tools.as_deref())?,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: Some(descriptor.id.to_string()),
        truncate_sequence: false,
    })
}

/// Builds a `mistralrs-core` normal request for a single embedding input.
pub(crate) fn map_embedding_request(
    prompt: String,
    descriptor: &ModelDescriptor,
    response: Sender<Response>,
    request_ordinal: usize,
) -> NormalRequest {
    NormalRequest {
        messages: RequestMessage::Embedding { prompt },
        sampling_params: SamplingParams::neutral(),
        response,
        return_logprobs: false,
        is_streaming: false,
        id: request_ordinal,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: Some(descriptor.id.to_string()),
        truncate_sequence: false,
    }
}

/// Builds a `mistralrs-core` tokenization request.
pub(crate) fn map_tokenization_request(
    request: &TokenizationRequest,
    descriptor: &ModelDescriptor,
    response: Sender<anyhow::Result<Vec<u32>>>,
) -> Result<MistralTokenizationRequest> {
    let text = match &request.input {
        nexo_core::TokenizationInput::Text(text) => Either::Right(text.clone()),
        nexo_core::TokenizationInput::Conversation(conversation) => {
            Either::Left(map_conversation(conversation, descriptor.role_strategy)?)
        }
    };

    Ok(MistralTokenizationRequest {
        text,
        tools: map_tool_definitions(&request.tools)?,
        add_generation_prompt: matches!(
            request.generation_prompt,
            nexo_core::GenerationPromptPolicy::Include
        ),
        add_special_tokens: matches!(request.special_tokens, SpecialTokenPolicy::Include),
        enable_thinking: Some(thinking_enabled(request.reasoning.thinking)),
        reasoning_effort: map_reasoning_effort(request.reasoning.effort),
        response,
    })
}

/// Builds a `mistralrs-core` detokenization request.
pub(crate) fn map_detokenization_request(
    request: &DetokenizationRequest,
    response: Sender<anyhow::Result<String>>,
) -> MistralDetokenizationRequest {
    MistralDetokenizationRequest {
        tokens: request.tokens.clone(),
        skip_special_tokens: matches!(request.special_tokens, SpecialTokenPolicy::Exclude),
        response,
    }
}

fn map_conversation(
    conversation: &Conversation,
    role_strategy: RoleStrategy,
) -> Result<Vec<IndexMap<String, MessageContent>>> {
    let mut mapped = Vec::new();
    for message in &conversation.messages {
        mapped.extend(map_message(message, role_strategy)?);
    }
    Ok(mapped)
}

fn map_message(
    message: &ConversationMessage,
    role_strategy: RoleStrategy,
) -> Result<Vec<IndexMap<String, MessageContent>>> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for part in &message.parts {
        match part {
            ContentPart::Text(TextPart { text }) => text_parts.push(text.clone()),
            ContentPart::ToolCall(call) => tool_calls.push(call.clone()),
            ContentPart::ToolResult(result) => tool_results.push(result.clone()),
            ContentPart::Image(_) => {
                return Err(Error::UnsupportedMessagePart {
                    part: "image input",
                });
            }
            ContentPart::Video(_) => {
                return Err(Error::UnsupportedMessagePart {
                    part: "video input",
                });
            }
            ContentPart::Audio(_) => {
                return Err(Error::UnsupportedMessagePart {
                    part: "audio input",
                });
            }
        }
    }

    if !tool_results.is_empty() {
        if !text_parts.is_empty() || !tool_calls.is_empty() || message.role != MessageRole::Tool {
            return Err(Error::UnsupportedMessagePart {
                part: "mixed tool-result message",
            });
        }

        return tool_results
            .iter()
            .map(map_tool_result_message)
            .collect::<Result<Vec<_>>>();
    }

    let mut mapped = IndexMap::new();
    mapped.insert(
        "role".to_string(),
        Either::Left(map_role(message.role, role_strategy).to_string()),
    );

    let content = if !text_parts.is_empty() {
        text_parts.join("\n")
    } else if !tool_calls.is_empty() {
        serialize_tool_calls(&tool_calls)?
    } else {
        String::new()
    };
    mapped.insert("content".to_string(), Either::Left(content));

    if !tool_calls.is_empty() {
        mapped.insert("tool_calls".to_string(), map_tool_calls_field(&tool_calls)?);
    }

    Ok(vec![mapped])
}

fn map_tool_result_message(
    result: &nexo_core::ToolResult,
) -> Result<IndexMap<String, MessageContent>> {
    let mut mapped = IndexMap::new();
    mapped.insert("role".to_string(), Either::Left("tool".to_string()));
    mapped.insert("name".to_string(), Either::Left(result.tool_name.clone()));
    mapped.insert(
        "content".to_string(),
        Either::Left(match &result.content {
            ToolResultContent::Text(text) => text.clone(),
            ToolResultContent::Json(value) => serde_json::to_string(value)?,
        }),
    );
    Ok(mapped)
}

fn map_sampling(config: &SamplingConfig) -> SamplingParams {
    let mut params = SamplingParams::neutral();
    params.temperature = config.temperature.map(f64::from);
    params.top_k = config.top_k.map(|value| value as usize);
    params.top_p = config.top_p.map(f64::from);
    params.min_p = config.min_p.map(f64::from);
    params.frequency_penalty = config.frequency_penalty;
    params.presence_penalty = config.presence_penalty;
    params.repetition_penalty = config.repetition_penalty;
    params.max_len = config.max_output_tokens;
    if !config.stop_sequences.is_empty() {
        params.stop_toks = Some(StopTokens::Seqs(config.stop_sequences.clone()));
    }
    params
}

fn map_constraint(constraint: &OutputConstraint) -> Constraint {
    match constraint {
        OutputConstraint::None => Constraint::None,
        OutputConstraint::JsonSchema(schema) => Constraint::JsonSchema(schema.clone()),
        OutputConstraint::Regex(regex) => Constraint::Regex(regex.clone()),
        OutputConstraint::LarkGrammar(grammar) => Constraint::Lark(grammar.clone()),
    }
}

fn map_tool_definitions(definitions: &[ToolDefinition]) -> Result<Option<Vec<Tool>>> {
    if definitions.is_empty() {
        return Ok(None);
    }

    definitions
        .iter()
        .map(|definition| {
            let parameters = match &definition.parameters {
                serde_json::Value::Null => None,
                serde_json::Value::Object(object) => Some(
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<HashMap<_, _>>(),
                ),
                other => {
                    return Err(Error::InvalidToolPayload {
                        tool_name: definition.name.clone(),
                        message: format!(
                            "expected an object or null for tool parameters, got {other}"
                        ),
                    });
                }
            };

            Ok(Tool {
                tp: ToolType::Function,
                function: MistralFunction {
                    description: Some(definition.description.clone()),
                    name: definition.name.clone(),
                    parameters,
                },
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn map_tool_choice(
    choice: &ToolChoice,
    tools: Option<&[Tool]>,
) -> Result<Option<MistralToolChoice>> {
    let Some(tools) = tools else {
        return Ok(None);
    };

    if tools.is_empty() {
        return Ok(None);
    }

    match choice {
        ToolChoice::Disabled => Ok(None),
        ToolChoice::Automatic => Ok(Some(MistralToolChoice::Auto)),
        ToolChoice::Specific { name } => {
            let tool = tools
                .iter()
                .find(|tool| tool.function.name == *name)
                .cloned()
                .ok_or_else(|| Error::InvalidToolPayload {
                    tool_name: name.clone(),
                    message: "forced tool choice was not present in the request tool list"
                        .to_string(),
                })?;
            Ok(Some(MistralToolChoice::Tool(tool)))
        }
    }
}

fn map_tool_calls_field(calls: &[ToolCall]) -> Result<MessageContent> {
    let mut mapped_calls = Vec::with_capacity(calls.len());

    for call in calls {
        let mut mapped_call = IndexMap::new();
        mapped_call.insert(
            "id".to_string(),
            serde_json::Value::String(call.id.to_string()),
        );
        mapped_call.insert(
            "type".to_string(),
            serde_json::Value::String("function".to_string()),
        );

        let arguments = call.arguments.clone();
        let mut function = serde_json::Map::new();
        function.insert(
            "name".to_string(),
            serde_json::Value::String(call.name.clone()),
        );
        function.insert("arguments".to_string(), arguments);
        mapped_call.insert("function".to_string(), serde_json::Value::Object(function));
        mapped_calls.push(mapped_call);
    }

    Ok(Either::Right(mapped_calls))
}

fn serialize_tool_calls(calls: &[ToolCall]) -> Result<String> {
    if calls.len() == 1 {
        return Ok(serde_json::to_string(&call_to_value(&calls[0]))?);
    }

    serde_json::to_string(
        &calls
            .iter()
            .map(call_to_value)
            .collect::<Vec<serde_json::Value>>(),
    )
    .map_err(Into::into)
}

fn call_to_value(call: &ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id.to_string(),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        }
    })
}

fn map_role(role: MessageRole, strategy: RoleStrategy) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::Developer => {
            if matches!(strategy, RoleStrategy::MergeDeveloperIntoSystem) {
                "system"
            } else {
                "developer"
            }
        }
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn thinking_enabled(mode: nexo_core::ThinkingMode) -> bool {
    matches!(mode, nexo_core::ThinkingMode::Enabled)
}

fn map_reasoning_effort(
    effort: Option<nexo_core::ReasoningEffort>,
) -> Option<MistralReasoningEffort> {
    effort.map(|effort| match effort {
        nexo_core::ReasoningEffort::Low => MistralReasoningEffort::Low,
        nexo_core::ReasoningEffort::Medium => MistralReasoningEffort::Medium,
        nexo_core::ReasoningEffort::High => MistralReasoningEffort::High,
    })
}

#[cfg(test)]
mod tests {
    use nexo_core::{
        ConversationMessage, MetadataMap, ModelCapability, ModelId, ModelModalities,
        ReasoningSettings, SupportedModality, ToolChoice, ToolExecutionConstraints,
        ToolParallelism, ToolSideEffectLevel,
    };

    use super::*;

    #[test]
    fn maps_text_generation_request_to_chat_request() {
        let (response, _receiver) = tokio::sync::mpsc::channel(1);
        let request = GenerateRequest {
            request_id: None,
            session_id: None,
            run_id: None,
            round_id: None,
            model: nexo_core::ModelSelection {
                specific_model: Some(ModelId::from("chat")),
                required_capabilities: Vec::new(),
                preferred_capabilities: Vec::new(),
            },
            conversation: Conversation {
                messages: vec![ConversationMessage {
                    role: MessageRole::Developer,
                    parts: vec![ContentPart::Text(TextPart {
                        text: "be concise".to_string(),
                    })],
                    metadata: MetadataMap::new(),
                }],
                metadata: MetadataMap::new(),
            },
            tools: Vec::new(),
            tool_choice: ToolChoice::Disabled,
            reasoning: ReasoningSettings::default(),
            output_constraint: OutputConstraint::None,
            sampling: SamplingConfig {
                max_output_tokens: Some(32),
                stop_sequences: vec!["stop".to_string()],
                ..SamplingConfig::default()
            },
            streaming: nexo_core::StreamingMode::Buffered,
            metadata: MetadataMap::new(),
        };
        let descriptor = descriptor(RoleStrategy::MergeDeveloperIntoSystem);

        let mapped = map_generate_request(&request, &descriptor, response, 1).unwrap();

        match mapped.messages {
            RequestMessage::Chat { messages, .. } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(
                    messages[0].get("role"),
                    Some(&Either::Left("system".to_string()))
                );
            }
            _ => panic!("expected chat request"),
        }
        assert_eq!(mapped.sampling_params.max_len, Some(32));
        assert!(matches!(
            mapped.sampling_params.stop_toks,
            Some(StopTokens::Seqs(_))
        ));
    }

    #[test]
    fn maps_tool_definition_and_forced_choice() {
        let definition = ToolDefinition {
            name: "lookup".to_string(),
            description: "Look up a value".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            contract_version: None,
            execution: ToolExecutionConstraints {
                timeout_ms: None,
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
            },
            metadata: MetadataMap::new(),
        };

        let tools = map_tool_definitions(std::slice::from_ref(&definition))
            .unwrap()
            .unwrap();
        let choice = map_tool_choice(
            &ToolChoice::Specific {
                name: "lookup".to_string(),
            },
            Some(&tools),
        )
        .unwrap();

        assert!(matches!(choice, Some(MistralToolChoice::Tool(_))));
    }

    fn descriptor(role_strategy: RoleStrategy) -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId::from("chat"),
            display_name: "chat".to_string(),
            provider: Some("test".to_string()),
            capabilities: vec![ModelCapability::TextGeneration],
            modalities: ModelModalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            role_strategy,
            context_window_tokens: Some(4096),
            max_output_tokens: Some(1024),
            metadata: MetadataMap::new(),
        }
    }
}
