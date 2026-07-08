use crate::Result;
use crate::engine::mistralrs::mapping::message::{MessageMapping, map_generate_message};
use crate::engine::mistralrs::mapping::tools::{map_tool_choice, map_tool_definitions};
use mistralrs_core::{
    Constraint, NormalRequest, ReasoningEffort as MistralReasoningEffort, RequestMessage, Response,
    SamplingParams, StopTokens,
};
use nexo_core::inference::requests::MultiModalPayload;
use nexo_core::{
    InferenceIntent, ModelDefinition, OutputConstraint, SamplingConfig, StreamingMode, ThinkingMode,
};
use tokio::sync::mpsc;

/// Maps a shared multimodal request into a Mistral.rs normal request.
///
/// # Arguments
///
/// * `request` - The full shared inference request carrying session identity.
/// * `payload` - The multimodal payload to translate into a Mistral.rs request.
/// * `descriptor` - The model definition used for role strategy and model identification.
/// * `response` - The one-shot response channel consumed by Mistral.rs.
/// * `request_ordinal` - The monotonically increasing request ordinal required by Mistral.rs.
pub(crate) fn map_multimodal_request(
    request: &InferenceIntent,
    payload: &MultiModalPayload,
    descriptor: &ModelDefinition,
    response: mpsc::Sender<Response>,
    request_ordinal: usize,
) -> Result<NormalRequest> {
    let tools = map_tool_definitions(&payload.tools)?;
    let mut mapped = MessageMapping::default();

    for message in &payload.conversation.messages {
        mapped.messages.extend(map_generate_message(
            message,
            *descriptor.role_strategy(),
            &mut mapped.images,
            &mut mapped.audios,
        )?);
    }

    let request_message = if mapped.images.is_empty() && mapped.audios.is_empty() {
        RequestMessage::Chat {
            messages: mapped.messages,
            enable_thinking: Some(thinking_enabled(payload.reasoning.thinking)),
            reasoning_effort: map_reasoning_effort(payload.reasoning.effort),
        }
    } else {
        RequestMessage::MultimodalChat {
            images: mapped.images,
            audios: mapped.audios,
            videos: Vec::new(),
            messages: mapped.messages,
            enable_thinking: Some(thinking_enabled(payload.reasoning.thinking)),
            reasoning_effort: map_reasoning_effort(payload.reasoning.effort),
        }
    };

    let mut normal_request = NormalRequest::new_simple(
        request_message,
        map_sampling(&payload.sampling),
        response,
        request_ordinal,
        tools.clone(),
        map_tool_choice(&payload.tool_choice, tools.as_deref())?,
    );
    normal_request.is_streaming = matches!(payload.streaming, StreamingMode::Streaming);
    normal_request.constraint = map_constraint(&payload.output_constraint);
    normal_request.model_id = Some(descriptor.id().to_string());
    normal_request.session_id = Some(request.session_id.to_string());

    Ok(normal_request)
}

/// Maps the shared sampling configuration into Mistral.rs sampling parameters.
///
/// # Arguments
///
/// * `config` - The shared sampling configuration requested by the caller.
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

/// Maps the shared output constraint into the equivalent Mistral.rs constraint.
///
/// # Arguments
///
/// * `constraint` - The structured output constraint requested by the caller.
fn map_constraint(constraint: &OutputConstraint) -> Constraint {
    match constraint {
        OutputConstraint::None => Constraint::None,
        OutputConstraint::JsonSchema(schema) => Constraint::JsonSchema(schema.clone()),
        OutputConstraint::Regex(regex) => Constraint::Regex(regex.clone()),
        OutputConstraint::LarkGrammar(grammar) => Constraint::Lark(grammar.clone()),
    }
}

/// Maps the shared thinking mode into the boolean flag expected by Mistral.rs.
///
/// # Arguments
///
/// * `mode` - The shared thinking mode requested by the caller.
fn thinking_enabled(mode: ThinkingMode) -> bool {
    matches!(mode, ThinkingMode::Enabled)
}

/// Maps the shared reasoning effort enum into the Mistral.rs reasoning effort enum.
///
/// # Arguments
///
/// * `effort` - The optional shared reasoning effort hint supplied with the request.
fn map_reasoning_effort(
    effort: Option<nexo_core::ReasoningEffort>,
) -> Option<MistralReasoningEffort> {
    effort.map(|effort| match effort {
        nexo_core::ReasoningEffort::Low => MistralReasoningEffort::Low,
        nexo_core::ReasoningEffort::Medium => MistralReasoningEffort::Medium,
        nexo_core::ReasoningEffort::High => MistralReasoningEffort::High,
    })
}
