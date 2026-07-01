use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, CompletionChunkResponse,
    CompletionResponse, Delta, Response, ResponseMessage, ResponseOk, ToolCallResponse,
};
use nexo_core::{
    ContentPart, ConversationMessage, FinishReason, InferenceFinal, InferenceMeta,
    InferenceOutput, InferenceUpdate, MessageRole, MultiModalDelta, MultiModalResponse,
    PerformanceMetrics, StreamSeq, TokenUsage, ToolCall, ToolCallDelta,
};

/// Maps a raw Mistral.rs response into the shared streamed inference update contract.
///
/// # Arguments
///
/// * `response` - The raw Mistral.rs response emitted for the current request.
/// * `meta` - Stable execution metadata emitted in each resulting update.
/// * `seq` - The sequence number assigned to progress updates.
pub(crate) fn map_multimodal_response(
    response: Response,
    meta: &InferenceMeta,
    seq: StreamSeq,
) -> nexo_core::Result<InferenceUpdate> {
    match response.as_result() {
        Ok(ResponseOk::Done(done)) => Ok(InferenceUpdate::completed(
            meta.clone(),
            InferenceFinal::MultiModal(map_chat_done(done)),
        )),
        Ok(ResponseOk::Chunk(chunk)) => Ok(InferenceUpdate::progress(
            meta.clone(),
            seq,
            InferenceOutput::MultiModal(map_chat_chunk(chunk)),
        )),
        Ok(ResponseOk::CompletionDone(done)) => Ok(InferenceUpdate::completed(
            meta.clone(),
            InferenceFinal::MultiModal(map_completion_done(done)),
        )),
        Ok(ResponseOk::CompletionChunk(chunk)) => Ok(InferenceUpdate::progress(
            meta.clone(),
            seq,
            InferenceOutput::MultiModal(map_completion_chunk(chunk)),
        )),
        Ok(ResponseOk::ImageGeneration(_)) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received image-generation output for a text generation request".to_string(),
        )),
        Ok(ResponseOk::Speech { .. }) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received speech output for a text generation request".to_string(),
        )),
        Ok(ResponseOk::Raw { .. }) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received raw logits output for a text generation request".to_string(),
        )),
        Ok(ResponseOk::Embeddings { .. }) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received embedding output for a text generation request".to_string(),
        )),
        Ok(ResponseOk::BlockDenoisingProgress(_)) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received block denoising progress for a text generation request".to_string(),
        )),
        Ok(ResponseOk::AgenticToolCallProgress { .. })
        | Ok(ResponseOk::AgenticToolApprovalRequired { .. })
        | Ok(ResponseOk::File(_)) => Ok(InferenceUpdate::failed(
            meta.clone(),
            "received unsupported agentic output for a text generation request".to_string(),
        )),
        Err(error) => Ok(InferenceUpdate::failed(meta.clone(), error.to_string())),
    }
}

/// Maps a completed chat response into the shared multimodal response contract.
///
/// # Arguments
///
/// * `done` - The completed chat response emitted by Mistral.rs.
fn map_chat_done(done: ChatCompletionResponse) -> MultiModalResponse {
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

    MultiModalResponse {
        message,
        reasoning,
        finish_reason,
        usage: Some(map_usage(&done.usage)),
        performance: Some(map_performance(&done.usage)),
    }
}

/// Maps a streaming chat chunk into the shared multimodal delta contract.
///
/// # Arguments
///
/// * `chunk` - The incremental chat chunk emitted by Mistral.rs.
fn map_chat_chunk(chunk: ChatCompletionChunkResponse) -> MultiModalDelta {
    let choice = chunk.choices.into_iter().next();
    let (mut delta, finish_reason) = if let Some(choice) = choice {
        (
            map_delta(choice.delta),
            choice
                .finish_reason
                .map(|reason| map_finish_reason(Some(&reason))),
        )
    } else {
        (MultiModalDelta::default(), None)
    };
    delta.usage = chunk.usage.as_ref().map(map_usage);
    delta.finish_reason = finish_reason;
    delta
}

/// Maps a completed completion-style response into the shared multimodal response contract.
///
/// # Arguments
///
/// * `done` - The completed completion response emitted by Mistral.rs.
fn map_completion_done(done: CompletionResponse) -> MultiModalResponse {
    let choice = done.choices.into_iter().next();
    let (message, finish_reason) = if let Some(choice) = choice {
        (
            ConversationMessage {
                role: MessageRole::Assistant,
                parts: vec![ContentPart::Text(choice.text)],
            },
            map_finish_reason(Some(&choice.finish_reason)),
        )
    } else {
        (empty_assistant_message(), FinishReason::Completed)
    };

    MultiModalResponse {
        message,
        reasoning: None,
        finish_reason,
        usage: Some(map_usage(&done.usage)),
        performance: Some(map_performance(&done.usage)),
    }
}

/// Maps a streaming completion chunk into the shared multimodal delta contract.
///
/// # Arguments
///
/// * `chunk` - The incremental completion chunk emitted by Mistral.rs.
fn map_completion_chunk(chunk: CompletionChunkResponse) -> MultiModalDelta {
    let choice = chunk.choices.into_iter().next();
    let (mut delta, finish_reason) = if let Some(choice) = choice {
        (
            MultiModalDelta {
                role: Some(MessageRole::Assistant),
                content_delta: Some(choice.text),
                reasoning_delta: None,
                tool_call_deltas: Vec::new(),
                usage: None,
                finish_reason: None,
            },
            choice
                .finish_reason
                .map(|reason| map_finish_reason(Some(&reason))),
        )
    } else {
        (MultiModalDelta::default(), None)
    };
    delta.finish_reason = finish_reason;
    delta
}

/// Maps a Mistral.rs response message into the shared conversation message contract.
///
/// # Arguments
///
/// * `message` - The Mistral.rs response message to translate.
fn map_response_message(message: ResponseMessage) -> ConversationMessage {
    let mut parts = Vec::new();
    if let Some(content) = message.content
        && !content.is_empty()
    {
        parts.push(ContentPart::Text(content));
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
    }
}

/// Maps a Mistral.rs streaming delta into the shared multimodal delta contract.
///
/// # Arguments
///
/// * `delta` - The Mistral.rs delta to translate.
fn map_delta(delta: Delta) -> MultiModalDelta {
    MultiModalDelta {
        role: Some(map_message_role(&delta.role)),
        content_delta: delta.content,
        reasoning_delta: delta.reasoning_content,
        tool_call_deltas: delta
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(map_tool_call_delta)
            .collect(),
        usage: None,
        finish_reason: None,
    }
}

/// Maps a full tool call response into the shared tool call representation.
///
/// # Arguments
///
/// * `response` - The tool call response emitted by Mistral.rs.
fn map_tool_call_response(response: ToolCallResponse) -> ToolCall {
    let arguments = serde_json::from_str(&response.function.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(response.function.arguments));

    ToolCall {
        id: nexo_core::ToolCallId::from(response.id),
        index: response.index,
        name: response.function.name,
        arguments,
    }
}

/// Maps a streaming tool call delta into the shared tool call delta representation.
///
/// # Arguments
///
/// * `response` - The tool call delta emitted by Mistral.rs.
fn map_tool_call_delta(response: ToolCallResponse) -> ToolCallDelta {
    ToolCallDelta {
        index: response.index,
        id: Some(nexo_core::ToolCallId::from(response.id)),
        name: Some(response.function.name),
        arguments_delta: Some(response.function.arguments),
    }
}

/// Maps a raw Mistral.rs message role string into the shared message role enum.
///
/// # Arguments
///
/// * `role` - The raw role string produced by Mistral.rs.
fn map_message_role(role: &str) -> MessageRole {
    match role {
        "system" => MessageRole::System,
        "developer" => MessageRole::Developer,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Assistant,
    }
}

/// Maps a raw Mistral.rs finish-reason string into the shared finish-reason enum.
///
/// # Arguments
///
/// * `reason` - The optional finish reason string emitted by Mistral.rs.
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

/// Maps Mistral.rs token accounting into the shared token-usage struct.
///
/// # Arguments
///
/// * `usage` - The token accounting metadata emitted by Mistral.rs.
fn map_usage(usage: &mistralrs_core::Usage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

/// Maps Mistral.rs timing metrics into the shared performance metrics struct.
///
/// # Arguments
///
/// * `usage` - The token and timing metadata emitted by Mistral.rs.
fn map_performance(usage: &mistralrs_core::Usage) -> PerformanceMetrics {
    PerformanceMetrics {
        total_duration_ms: (usage.total_time_sec * 1000.0).round() as u64,
        input_tokens_per_second: Some(usage.avg_prompt_tok_per_sec),
        output_tokens_per_second: Some(usage.avg_compl_tok_per_sec),
    }
}

/// Creates an empty assistant message for responses that complete without content.
fn empty_assistant_message() -> ConversationMessage {
    ConversationMessage {
        role: MessageRole::Assistant,
        parts: Vec::new(),
    }
}
