use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, CompletionChunkResponse,
    CompletionResponse, Delta, Response, ResponseErr, ResponseMessage, ResponseOk,
    ToolCallResponse, Usage,
};
use nexo_core::{
    ContentPart, ConversationMessage, FinishReason, GenerateChunk, GenerateCompleted,
    GenerateDelta, GenerateStarted, InferenceErrorCode, InferenceFailure, InferenceResponse,
    MessageRole, PerformanceMetrics, RequestId, Retryability, RoundId, RunId, TextPart, TokenUsage,
    ToolCall, ToolCallDelta, ToolCallId,
};

use crate::Error;

/// Stable response metadata captured when a request is submitted.
#[derive(Debug, Clone)]
pub(crate) struct ResponseContext {
    /// The caller-supplied request identifier.
    pub request_id: Option<RequestId>,

    /// The caller-supplied run identifier.
    pub run_id: Option<RunId>,

    /// The caller-supplied round identifier.
    pub round_id: Option<RoundId>,

    /// The concrete model identifier selected for the request.
    pub model_id: nexo_core::ModelId,
}

/// Creates the initial `generation_started` event for a request.
pub(crate) fn generation_started(context: &ResponseContext) -> InferenceResponse {
    InferenceResponse::GenerationStarted(GenerateStarted {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        model_id: Some(context.model_id.clone()),
    })
}

/// Maps a `mistralrs-core` generation response into a `nexo-core` response.
pub(crate) fn map_generation_response(
    response: Response,
    context: &ResponseContext,
) -> InferenceResponse {
    match response.as_result() {
        Ok(ResponseOk::Done(done)) => {
            InferenceResponse::GenerationCompleted(map_chat_done(done, context))
        }
        Ok(ResponseOk::Chunk(chunk)) => {
            InferenceResponse::GenerationChunk(map_chat_chunk(chunk, context))
        }
        Ok(ResponseOk::CompletionDone(done)) => {
            InferenceResponse::GenerationCompleted(map_completion_done(done, context))
        }
        Ok(ResponseOk::CompletionChunk(chunk)) => {
            InferenceResponse::GenerationChunk(map_completion_chunk(chunk, context))
        }
        Ok(ResponseOk::ImageGeneration(_)) => failure_response(
            InferenceErrorCode::UnsupportedFeature,
            "received image-generation output for a text generation request".to_string(),
            Retryability::Fatal,
            context,
        ),
        Ok(ResponseOk::Speech { .. }) => failure_response(
            InferenceErrorCode::UnsupportedFeature,
            "received speech output for a text generation request".to_string(),
            Retryability::Fatal,
            context,
        ),
        Ok(ResponseOk::Raw { .. }) => failure_response(
            InferenceErrorCode::Internal,
            "received raw logits output for a text generation request".to_string(),
            Retryability::Fatal,
            context,
        ),
        Ok(ResponseOk::Embeddings { .. }) => failure_response(
            InferenceErrorCode::UnsupportedFeature,
            "received embedding output for a text generation request".to_string(),
            Retryability::Fatal,
            context,
        ),
        Err(error) => map_response_error(*error, context),
    }
}

/// Maps an embedding response into the shared response enum.
pub(crate) fn map_embedding_response(
    request_id: Option<RequestId>,
    model_id: nexo_core::ModelId,
    vectors: Vec<nexo_core::EmbeddingVector>,
    usage: Option<TokenUsage>,
) -> InferenceResponse {
    InferenceResponse::Embeddings(nexo_core::EmbeddingResponse {
        request_id,
        model_id: Some(model_id),
        vectors,
        usage,
    })
}

/// Creates a structured inference failure from a crate-local error.
pub(crate) fn map_runtime_error(
    error: Error,
    request_id: Option<RequestId>,
    run_id: Option<RunId>,
    round_id: Option<RoundId>,
) -> InferenceResponse {
    let (code, retryability) = match error {
        Error::UnsupportedFeature { .. }
        | Error::UnsupportedRequest { .. }
        | Error::UnsupportedMessagePart { .. } => {
            (InferenceErrorCode::UnsupportedFeature, Retryability::Fatal)
        }
        Error::UnknownModel { .. }
        | Error::UnresolvedModelSelection { .. }
        | Error::InvalidToolPayload { .. }
        | Error::Json(_) => (InferenceErrorCode::InvalidRequest, Retryability::Fatal),
        Error::MistralRuntime { .. }
        | Error::Io(_)
        | Error::Config { .. }
        | Error::EmptyModelCatalog
        | Error::DuplicateModelId { .. } => (InferenceErrorCode::Internal, Retryability::Retryable),
        Error::Core(core_error) => {
            return InferenceResponse::Failure(InferenceFailure {
                request_id,
                run_id,
                round_id,
                code: match core_error {
                    nexo_core::Error::InvalidRequest { .. } => InferenceErrorCode::InvalidRequest,
                    nexo_core::Error::UnsupportedFeature { .. } => {
                        InferenceErrorCode::UnsupportedFeature
                    }
                    nexo_core::Error::InvalidState { .. } => InferenceErrorCode::Internal,
                },
                message: core_error.to_string(),
                retryability: Retryability::Fatal,
            });
        }
    };

    InferenceResponse::Failure(InferenceFailure {
        request_id,
        run_id,
        round_id,
        code,
        message: error.to_string(),
        retryability,
    })
}

fn map_chat_done(done: ChatCompletionResponse, context: &ResponseContext) -> GenerateCompleted {
    let reasoning = done
        .choices
        .first()
        .and_then(|choice| choice.message.reasoning_content.clone());
    let choice = done.choices.into_iter().next();
    let finish_reason = choice
        .as_ref()
        .map(|choice| map_finish_reason(Some(choice.finish_reason.as_str())))
        .unwrap_or(FinishReason::Completed);

    let message = choice
        .map(|choice| map_response_message(choice.message))
        .unwrap_or_else(empty_assistant_message);

    GenerateCompleted {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        model_id: Some(context.model_id.clone()),
        reasoning,
        message,
        finish_reason,
        usage: Some(map_usage(&done.usage)),
        performance: Some(map_performance(&done.usage)),
    }
}

fn map_chat_chunk(chunk: ChatCompletionChunkResponse, context: &ResponseContext) -> GenerateChunk {
    let choice = chunk.choices.into_iter().next();
    let (delta, finish_reason) = if let Some(choice) = choice {
        (
            map_delta(choice.delta),
            choice
                .finish_reason
                .map(|reason| map_finish_reason(Some(&reason))),
        )
    } else {
        (GenerateDelta::default(), None)
    };

    GenerateChunk {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        model_id: Some(context.model_id.clone()),
        delta,
        usage: chunk.usage.as_ref().map(map_usage),
        finish_reason,
    }
}

fn map_completion_done(done: CompletionResponse, context: &ResponseContext) -> GenerateCompleted {
    let choice = done.choices.into_iter().next();
    let (message, finish_reason) = if let Some(choice) = choice {
        (
            ConversationMessage {
                role: MessageRole::Assistant,
                parts: vec![ContentPart::Text(TextPart { text: choice.text })],
                metadata: nexo_core::MetadataMap::new(),
            },
            map_finish_reason(Some(&choice.finish_reason)),
        )
    } else {
        (empty_assistant_message(), FinishReason::Completed)
    };

    GenerateCompleted {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        model_id: Some(context.model_id.clone()),
        message,
        reasoning: None,
        finish_reason,
        usage: Some(map_usage(&done.usage)),
        performance: Some(map_performance(&done.usage)),
    }
}

fn map_completion_chunk(
    chunk: CompletionChunkResponse,
    context: &ResponseContext,
) -> GenerateChunk {
    let choice = chunk.choices.into_iter().next();
    let (delta, finish_reason) = if let Some(choice) = choice {
        (
            GenerateDelta {
                role: Some(MessageRole::Assistant),
                content_delta: Some(choice.text),
                reasoning_delta: None,
                tool_call_deltas: Vec::new(),
            },
            choice
                .finish_reason
                .map(|reason| map_finish_reason(Some(&reason))),
        )
    } else {
        (GenerateDelta::default(), None)
    };

    GenerateChunk {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        model_id: Some(context.model_id.clone()),
        delta,
        usage: None,
        finish_reason,
    }
}

fn map_response_message(message: ResponseMessage) -> ConversationMessage {
    let mut parts = Vec::new();
    if let Some(content) = message.content
        && !content.is_empty()
    {
        parts.push(ContentPart::Text(TextPart { text: content }));
    }
    if let Some(tool_calls) = message.tool_calls {
        parts.extend(
            tool_calls
                .into_iter()
                .map(map_tool_call_response)
                .map(ContentPart::ToolCall),
        );
    }

    ConversationMessage {
        role: map_message_role(&message.role),
        parts,
        metadata: nexo_core::MetadataMap::new(),
    }
}

fn map_delta(delta: Delta) -> GenerateDelta {
    GenerateDelta {
        role: Some(map_message_role(&delta.role)),
        content_delta: delta.content,
        reasoning_delta: delta.reasoning_content,
        tool_call_deltas: delta
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(map_tool_call_delta)
            .collect(),
    }
}

fn map_tool_call_response(response: ToolCallResponse) -> ToolCall {
    let arguments = serde_json::from_str(&response.function.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(response.function.arguments));

    ToolCall {
        id: ToolCallId::from(response.id),
        index: response.index,
        name: response.function.name,
        arguments,
    }
}

fn map_tool_call_delta(response: ToolCallResponse) -> ToolCallDelta {
    ToolCallDelta {
        index: response.index,
        id: Some(ToolCallId::from(response.id)),
        name: Some(response.function.name),
        arguments_delta: Some(response.function.arguments),
    }
}

fn map_message_role(role: &str) -> MessageRole {
    match role {
        "system" => MessageRole::System,
        "developer" => MessageRole::Developer,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Assistant,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("length") => FinishReason::MaxTokens,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("cancelled") => FinishReason::Cancelled,
        Some("content_filter") => FinishReason::ContentFiltered,
        Some("stop") | Some("eos") | None => FinishReason::Completed,
        Some(_) => FinishReason::Completed,
    }
}

fn map_usage(usage: &Usage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

fn map_performance(usage: &Usage) -> PerformanceMetrics {
    PerformanceMetrics {
        total_duration_ms: (usage.total_time_sec * 1000.0).round() as u64,
        input_tokens_per_second: Some(usage.avg_prompt_tok_per_sec),
        output_tokens_per_second: Some(usage.avg_compl_tok_per_sec),
    }
}

fn map_response_error(error: ResponseErr, context: &ResponseContext) -> InferenceResponse {
    let (code, retryability) = match error {
        ResponseErr::ValidationError(_) => {
            (InferenceErrorCode::InvalidRequest, Retryability::Fatal)
        }
        ResponseErr::InternalError(_) => (InferenceErrorCode::Internal, Retryability::Retryable),
        ResponseErr::ModelError(_, _) | ResponseErr::CompletionModelError(_, _) => {
            (InferenceErrorCode::Internal, Retryability::Retryable)
        }
    };

    InferenceResponse::Failure(InferenceFailure {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        code,
        message: error.to_string(),
        retryability,
    })
}

fn failure_response(
    code: InferenceErrorCode,
    message: String,
    retryability: Retryability,
    context: &ResponseContext,
) -> InferenceResponse {
    InferenceResponse::Failure(InferenceFailure {
        request_id: context.request_id.clone(),
        run_id: context.run_id.clone(),
        round_id: context.round_id.clone(),
        code,
        message,
        retryability,
    })
}

fn empty_assistant_message() -> ConversationMessage {
    ConversationMessage {
        role: MessageRole::Assistant,
        parts: Vec::new(),
        metadata: nexo_core::MetadataMap::new(),
    }
}
