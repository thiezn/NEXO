use std::path::{Path, PathBuf};

use chrono::Local;
use nexo_ws_schema::{
    AudioAnalyzeResponse, AudioGenerateResponse, CronListParams, CronListResponse, CronPayload,
    EventKind, Frame, HealthParams, HealthResponse, ImageAnalyzeResponse, ImageGenerateResponse,
    MessagePayload, Method, PresencePayload, PromptCollectionListParams,
    PromptCollectionListResponse, PromptDocumentListParams, PromptDocumentListResponse,
    RunEventPayload, RunStartResponse, RunStatus, SendResponse, SessionClearResponse,
    SessionClosedPayload, SessionCreateResponse, SessionGetResponse, SessionListParams,
    SessionListResponse, ShutdownPayload, StatusParams, StatusResponse, ToolEntry,
    ToolsCatalogParams, ToolsCatalogResponse, ToolsExecuteResponse,
};
use super::command::{self, AppCommand, CommandContext};
use super::message::Message;
use super::model::{ActiveStream, ActivityButton, LogKind, Model, PendingRequest, RunningState};
use super::network::NetworkEvent;

#[derive(Debug)]
pub enum Effect {
    Send(Frame),
    CopyToClipboard { label: &'static str, text: String },
    Close,
}

pub fn update(model: &mut Model, message: Message) -> Vec<Effect> {
    match message {
        Message::Tick => handle_tick(model),
        Message::Network(event) => handle_network_event(model, event),
        Message::SubmitInput => submit_input(model),
        Message::ShowHelp(visible) => {
            model.show_help = visible;
            Vec::new()
        }
        Message::InsertChar(ch) => {
            model.insert_char(ch);
            Vec::new()
        }
        Message::Backspace => {
            model.backspace();
            Vec::new()
        }
        Message::Delete => {
            model.delete();
            Vec::new()
        }
        Message::MoveCursorLeft => {
            model.move_cursor_left();
            Vec::new()
        }
        Message::MoveCursorRight => {
            model.move_cursor_right();
            Vec::new()
        }
        Message::MoveCursorHome => {
            model.move_cursor_home();
            Vec::new()
        }
        Message::MoveCursorEnd => {
            model.move_cursor_end();
            Vec::new()
        }
        Message::AcceptCompletion => {
            model.accept_completion();
            Vec::new()
        }
        Message::SelectNextCompletion => {
            model.select_next_completion();
            Vec::new()
        }
        Message::SelectPrevCompletion => {
            model.select_prev_completion();
            Vec::new()
        }
        Message::ClearCompletion => {
            model.clear_completion();
            model.show_help = false;
            Vec::new()
        }
        Message::Click { column, row } => handle_click(model, column, row),
        Message::ScrollActivityUp { column, row } => {
            if model.show_help {
                return Vec::new();
            }
            model.scroll_activity_up(column, row, 3);
            Vec::new()
        }
        Message::ScrollActivityDown { column, row } => {
            if model.show_help {
                return Vec::new();
            }
            model.scroll_activity_down(column, row, 3);
            Vec::new()
        }
        Message::Quit => {
            model.running_state = RunningState::Done;
            vec![Effect::Close]
        }
    }
}

fn handle_click(model: &mut Model, column: u16, row: u16) -> Vec<Effect> {
    if model.show_help {
        return Vec::new();
    }

    match model.activity_button_at(column, row) {
        Some(ActivityButton::CopyAll) => vec![Effect::CopyToClipboard {
            label: "all activity",
            text: model.all_activity_text(),
        }],
        Some(ActivityButton::CopyLastOutput) => match model.last_output_text() {
            Some(text) if !text.is_empty() => vec![Effect::CopyToClipboard {
                label: "last output",
                text,
            }],
            _ => {
                model.push_log(LogKind::Warning, "copy", "No output available to copy");
                Vec::new()
            }
        },
        None => Vec::new(),
    }
}

fn handle_tick(model: &mut Model) -> Vec<Effect> {
    let mut effects = Vec::new();

    if let Some(session_name) = model.startup_session_name.take() {
        match enqueue_request(
            model,
            PendingRequest::SessionCreate,
            Method::SessionCreate,
            nexo_ws_schema::SessionCreateParams {
                name: Some(session_name),
                prompt_collection_id: None,
            },
        ) {
            Ok(effect) => effects.push(effect),
            Err(error) => model.push_log(LogKind::Error, "session create", error),
        }
    }

    if model.should_refresh_status() {
        match enqueue_request(
            model,
            PendingRequest::Status { silent: true },
            Method::Status,
            StatusParams::default(),
        ) {
            Ok(effect) => effects.push(effect),
            Err(error) => model.push_log(LogKind::Error, "status", error),
        }
    }

    effects
}

fn submit_input(model: &mut Model) -> Vec<Effect> {
    let input = model.take_input();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let context = CommandContext {
        current_session_id: model.current_session_id.as_deref(),
        default_session_name: model.default_session_name.as_deref(),
        default_model_id: model.default_model_id.as_deref(),
        workspace_root: &model.workspace_root,
    };

    let command = match command::parse(trimmed, context) {
        Ok(command) => command,
        Err(error) => {
            model.push_log(LogKind::Error, "Command", error);
            return Vec::new();
        }
    };

    model.push_log(LogKind::Command, "Input", trimmed);
    execute_command(model, command)
}

fn execute_command(model: &mut Model, command: AppCommand) -> Vec<Effect> {
    match command {
        AppCommand::Help => {
            model.show_help = true;
            Vec::new()
        }
        AppCommand::Quit => {
            model.running_state = RunningState::Done;
            vec![Effect::Close]
        }
        AppCommand::Clear => {
            model.clear_logs();
            model.push_log(LogKind::Info, "Logs", "Cleared log output");
            Vec::new()
        }
        AppCommand::Health => single_request(
            model,
            PendingRequest::Health,
            Method::Health,
            HealthParams::default(),
        ),
        AppCommand::Status => single_request(
            model,
            PendingRequest::Status { silent: false },
            Method::Status,
            StatusParams::default(),
        ),
        AppCommand::Send(params) => {
            single_request(model, PendingRequest::Send, Method::Send, params)
        }
        AppCommand::RunStart(params) => {
            model.active_stream = None;
            single_request(model, PendingRequest::RunStart, Method::RunStart, params)
        }
        AppCommand::SessionCreate(params) => single_request(
            model,
            PendingRequest::SessionCreate,
            Method::SessionCreate,
            params,
        ),
        AppCommand::SessionList => single_request(
            model,
            PendingRequest::SessionList,
            Method::SessionList,
            SessionListParams::default(),
        ),
        AppCommand::SessionGet(params) => single_request(
            model,
            PendingRequest::SessionGet,
            Method::SessionGet,
            params,
        ),
        AppCommand::SessionClear(params) => single_request(
            model,
            PendingRequest::SessionClear,
            Method::SessionClear,
            params,
        ),
        AppCommand::ToolsCatalog => single_request(
            model,
            PendingRequest::ToolsCatalog,
            Method::ToolsCatalog,
            ToolsCatalogParams::default(),
        ),
        AppCommand::ToolsExecute(params) => single_request(
            model,
            PendingRequest::ToolsExecute,
            Method::ToolsExecute,
            params,
        ),
        AppCommand::CronCreate(params) => single_request(
            model,
            PendingRequest::CronCreate,
            Method::CronCreate,
            params,
        ),
        AppCommand::CronList => single_request(
            model,
            PendingRequest::CronList,
            Method::CronList,
            CronListParams::default(),
        ),
        AppCommand::CronDelete(params) => single_request(
            model,
            PendingRequest::CronDelete,
            Method::CronDelete,
            params,
        ),
        AppCommand::PromptDocumentCreate(params) => single_request(
            model,
            PendingRequest::PromptDocumentCreate,
            Method::PromptDocumentCreate,
            params,
        ),
        AppCommand::PromptDocumentList => single_request(
            model,
            PendingRequest::PromptDocumentList,
            Method::PromptDocumentList,
            PromptDocumentListParams::default(),
        ),
        AppCommand::PromptDocumentDelete(params) => single_request(
            model,
            PendingRequest::PromptDocumentDelete,
            Method::PromptDocumentDelete,
            params,
        ),
        AppCommand::PromptCollectionCreate(params) => single_request(
            model,
            PendingRequest::PromptCollectionCreate,
            Method::PromptCollectionCreate,
            params,
        ),
        AppCommand::PromptCollectionList => single_request(
            model,
            PendingRequest::PromptCollectionList,
            Method::PromptCollectionList,
            PromptCollectionListParams::default(),
        ),
        AppCommand::PromptCollectionDelete(params) => single_request(
            model,
            PendingRequest::PromptCollectionDelete,
            Method::PromptCollectionDelete,
            params,
        ),
        AppCommand::SystemPresence(params) => single_request(
            model,
            PendingRequest::SystemPresence,
            Method::SystemPresence,
            params,
        ),
        AppCommand::ImageAnalyze(params) => single_request(
            model,
            PendingRequest::ImageAnalyze,
            Method::ImageAnalyze,
            params,
        ),
        AppCommand::AudioAnalyze(params) => single_request(
            model,
            PendingRequest::AudioAnalyze,
            Method::AudioAnalyze,
            params,
        ),
        AppCommand::ImageGenerate(params) => single_request(
            model,
            PendingRequest::ImageGenerate,
            Method::ImageGenerate,
            params,
        ),
        AppCommand::AudioGenerate(params) => single_request(
            model,
            PendingRequest::AudioGenerate,
            Method::AudioGenerate,
            params,
        ),
    }
}

fn handle_network_event(model: &mut Model, event: NetworkEvent) -> Vec<Effect> {
    match event {
        NetworkEvent::Frame(frame) => handle_frame(model, frame),
        NetworkEvent::Disconnected(error) => {
            model.set_disconnected(error);
            Vec::new()
        }
    }
}

fn handle_frame(model: &mut Model, frame: Frame) -> Vec<Effect> {
    match frame {
        Frame::Response {
            id,
            ok,
            payload,
            error,
        } => {
            let Some(pending) = model.pending_requests.remove(&id) else {
                model.push_log(
                    LogKind::Warning,
                    "Response",
                    format!("Received response for unknown request id {id}"),
                );
                return Vec::new();
            };

            if !ok {
                let error = error
                    .map(|error| format!("{}: {}", error.code, error.message))
                    .unwrap_or_else(|| "Unknown gateway error".to_string());
                model.push_log(LogKind::Error, pending.label(), error);
                return Vec::new();
            }

            handle_success_response(model, pending, payload.unwrap_or(serde_json::Value::Null))
        }
        Frame::Event { event, payload, .. } => handle_event(model, event, payload),
        Frame::Request { method, .. } => {
            model.push_log(
                LogKind::Warning,
                "Gateway request",
                format!("Ignoring server-initiated request for method {method:?}"),
            );
            Vec::new()
        }
    }
}

fn handle_success_response(
    model: &mut Model,
    pending: PendingRequest,
    payload: serde_json::Value,
) -> Vec<Effect> {
    match pending {
        PendingRequest::Health => {
            let Some(response) = decode_response::<HealthResponse>(model, "health", payload) else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Success,
                "health",
                format!(
                    "Gateway is {} (uptime: {}s)",
                    response.status, response.uptime_secs
                ),
            );
        }
        PendingRequest::Status { silent } => {
            let Some(response) = decode_response::<StatusResponse>(model, "status", payload) else {
                return Vec::new();
            };
            model.summary.connected_nodes = Some(response.connected_nodes);
            model.summary.connected_clients = Some(response.connected_users);
            model.summary.capabilities = response.capabilities.clone();
            if !silent {
                model.push_log(
                    LogKind::Response,
                    "status",
                    format!(
                        "connected=true\nclients={}\nnodes={}\ncapabilities={}",
                        response.connected_users,
                        response.connected_nodes,
                        response.capabilities.join(", ")
                    ),
                );
            }
        }
        PendingRequest::Send => {
            let Some(response) = decode_response::<SendResponse>(model, "send", payload) else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Success,
                "send",
                format!("delivered={}", response.delivered),
            );
        }
        PendingRequest::RunStart => {
            let Some(response) = decode_response::<RunStartResponse>(model, "run", payload) else {
                return Vec::new();
            };
            model.current_session_id = Some(response.session_id.clone());
            model.active_stream = Some(ActiveStream {
                run_id: response.run_id.clone(),
                session_id: response.session_id.clone(),
                status: response.status,
                content: String::new(),
                tool_name: None,
                tool_call_id: None,
                thinking_content: None,
                error: None,
            });
            model.push_log(
                LogKind::Info,
                "run",
                format!(
                    "run={} session={} status={:?}",
                    response.run_id, response.session_id, response.status
                ),
            );
            if let Some(summary) = response.summary {
                model.push_log(LogKind::Info, "run summary", summary);
            }
        }
        PendingRequest::SessionCreate => {
            let Some(response) =
                decode_response::<SessionCreateResponse>(model, "session create", payload)
            else {
                return Vec::new();
            };
            model.current_session_id = Some(response.session_id.clone());
            model.push_log(
                LogKind::Success,
                "session create",
                format!("Current session is {}", response.session_id),
            );
        }
        PendingRequest::SessionList => {
            let Some(response) =
                decode_response::<SessionListResponse>(model, "session list", payload)
            else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Response,
                "session list",
                format_session_list(&response),
            );
        }
        PendingRequest::SessionGet => {
            let Some(response) =
                decode_response::<SessionGetResponse>(model, "session get", payload)
            else {
                return Vec::new();
            };
            model.current_session_id = Some(response.session_id.clone());
            model.push_log(
                LogKind::Response,
                "session get",
                format_session_get(&response),
            );
        }
        PendingRequest::SessionClear => {
            let Some(response) =
                decode_response::<SessionClearResponse>(model, "session clear", payload)
            else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Success,
                "session clear",
                format!("cleared={}", response.cleared),
            );
        }
        PendingRequest::ToolsCatalog => {
            let Some(response) =
                decode_response::<ToolsCatalogResponse>(model, "tools catalog", payload)
            else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Response,
                "tools catalog",
                format_tools_catalog(&response.tools),
            );
        }
        PendingRequest::ToolsExecute => {
            let Some(response) =
                decode_response::<ToolsExecuteResponse>(model, "tools execute", payload)
            else {
                return Vec::new();
            };
            let mut parts = vec![format!("success={}", response.success)];
            if !response.output.is_empty() {
                parts.push(response.output);
            }
            if let Some(error) = response.error {
                parts.push(format!("error={error}"));
            }
            model.push_log(
                if response.success {
                    LogKind::Success
                } else {
                    LogKind::Error
                },
                "tools execute",
                parts.join("\n"),
            );
        }
        PendingRequest::CronList => {
            let Some(response) = decode_response::<CronListResponse>(model, "cron list", payload)
            else {
                return Vec::new();
            };
            model.push_log(LogKind::Response, "cron list", format_cron_list(&response));
        }
        PendingRequest::PromptDocumentList => {
            let Some(response) = decode_response::<PromptDocumentListResponse>(
                model,
                "prompt document list",
                payload,
            ) else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Response,
                "prompt document list",
                format_prompt_document_list(&response),
            );
        }
        PendingRequest::PromptCollectionList => {
            let Some(response) = decode_response::<PromptCollectionListResponse>(
                model,
                "prompt collection list",
                payload,
            ) else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Response,
                "prompt collection list",
                format_prompt_collection_list(&response),
            );
        }
        PendingRequest::ImageAnalyze => {
            let Some(response) =
                decode_response::<ImageAnalyzeResponse>(model, "analyze image", payload)
            else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Success,
                "analyze image",
                format!(
                    "{}\n\n[{} tokens in {}ms]",
                    response.text, response.tokens_generated, response.inference_time_ms
                ),
            );
        }
        PendingRequest::AudioAnalyze => {
            let Some(response) =
                decode_response::<AudioAnalyzeResponse>(model, "analyze audio", payload)
            else {
                return Vec::new();
            };
            model.push_log(
                LogKind::Success,
                "analyze audio",
                format!(
                    "{}\n\n[{} tokens in {}ms]",
                    response.text, response.tokens_generated, response.inference_time_ms
                ),
            );
        }
        PendingRequest::ImageGenerate => {
            let Some(response) =
                decode_response::<ImageGenerateResponse>(model, "generate image", payload)
            else {
                return Vec::new();
            };
            handle_image_generate_response(model, response);
        }
        PendingRequest::AudioGenerate => {
            let Some(response) =
                decode_response::<AudioGenerateResponse>(model, "generate audio", payload)
            else {
                return Vec::new();
            };
            handle_audio_generate_response(model, response);
        }
        other => {
            model.push_log(LogKind::Response, other.label(), pretty_json(&payload));
        }
    }

    Vec::new()
}

fn handle_event(model: &mut Model, event: EventKind, payload: serde_json::Value) -> Vec<Effect> {
    match event {
        EventKind::Run => handle_run_event(model, payload),
        EventKind::Message => {
            match serde_json::from_value::<MessagePayload>(payload) {
                Ok(message) => {
                    model.push_log(
                        LogKind::Event,
                        format!("message from {}", message.from),
                        pretty_json(&message.payload),
                    );
                }
                Err(error) => model.push_log(
                    LogKind::Error,
                    "message",
                    format!("Invalid message event payload: {error}"),
                ),
            }
            Vec::new()
        }
        EventKind::Presence => {
            match serde_json::from_value::<PresencePayload>(payload) {
                Ok(presence) => {
                    model.last_status_request_at = None;
                    model.push_log(
                        LogKind::Event,
                        "presence",
                        format!(
                            "client={} role={:?} status={}",
                            presence.client_id, presence.role, presence.status
                        ),
                    );
                }
                Err(error) => model.push_log(
                    LogKind::Error,
                    "presence",
                    format!("Invalid presence event payload: {error}"),
                ),
            }
            Vec::new()
        }
        EventKind::Shutdown => {
            match serde_json::from_value::<ShutdownPayload>(payload) {
                Ok(shutdown) => {
                    model.summary.connected = false;
                    model.push_log(LogKind::Warning, "shutdown", shutdown.reason);
                }
                Err(error) => model.push_log(
                    LogKind::Error,
                    "shutdown",
                    format!("Invalid shutdown event payload: {error}"),
                ),
            }
            Vec::new()
        }
        EventKind::Cron => {
            match serde_json::from_value::<CronPayload>(payload) {
                Ok(cron) => model.push_log(
                    LogKind::Event,
                    "cron",
                    format!("job={} name={}", cron.job_id, cron.name),
                ),
                Err(error) => model.push_log(
                    LogKind::Error,
                    "cron",
                    format!("Invalid cron event payload: {error}"),
                ),
            }
            Vec::new()
        }
        EventKind::SessionClosed => {
            match serde_json::from_value::<SessionClosedPayload>(payload) {
                Ok(session_closed) => {
                    if model.current_session_id.as_deref() == Some(&session_closed.session_id) {
                        model.current_session_id = None;
                    }
                    model.push_log(
                        LogKind::Event,
                        "session closed",
                        format!("session_id={}", session_closed.session_id),
                    );
                }
                Err(error) => model.push_log(
                    LogKind::Error,
                    "session closed",
                    format!("Invalid session closed event payload: {error}"),
                ),
            }
            Vec::new()
        }
        EventKind::Tick | EventKind::Heartbeat => Vec::new(),
    }
}

fn handle_run_event(model: &mut Model, payload: serde_json::Value) -> Vec<Effect> {
    let event: RunEventPayload = match serde_json::from_value(payload) {
        Ok(event) => event,
        Err(error) => {
            model.push_log(
                LogKind::Error,
                "run",
                format!("Invalid event payload: {error}"),
            );
            return Vec::new();
        }
    };

    let stream = model.active_stream.get_or_insert_with(|| ActiveStream {
        run_id: event.run_id.clone(),
        session_id: event.session_id.clone(),
        status: event.status,
        content: String::new(),
        tool_name: None,
        tool_call_id: None,
        thinking_content: None,
        error: None,
    });

    if stream.run_id != event.run_id {
        model.push_log(
            LogKind::Warning,
            "agent",
            format!("Ignoring event for inactive run {}", event.run_id),
        );
        return Vec::new();
    }

    stream.session_id = event.session_id.clone();
    stream.status = event.status;
    stream.tool_name = event.tool_name.clone();
    stream.tool_call_id = event.tool_call_id.clone();
    stream.error = event.error.clone();
    if let Some(thinking_content) = event.thinking_content.clone() {
        stream.thinking_content = Some(thinking_content);
    }
    if matches!(event.status, RunStatus::Streaming | RunStatus::Accepted)
        && let Some(content) = &event.content
    {
        stream.content = content.clone();
    }

    match event.status {
        RunStatus::Queued => model.push_log(
            LogKind::Info,
            "agent",
            event
                .content
                .unwrap_or_else(|| "Run queued, waiting for an inference node".to_string()),
        ),
        RunStatus::Thinking => {}
        RunStatus::ToolCall => {
            let tool_name = format_tool_activity(
                event.tool_name.as_deref().unwrap_or("unknown"),
                event.tool_call_id.as_deref(),
            );
            if let Some(content) = event.content {
                model.push_log(
                    LogKind::Info,
                    "agent tool result",
                    format!("{tool_name}\n{content}"),
                );
            } else {
                model.push_log(LogKind::Info, "agent tool", tool_name);
            }
        }
        RunStatus::Streaming | RunStatus::Accepted => {}
        RunStatus::Completed => {
            if let Some(stream) = model.active_stream.take() {
                model.current_session_id = Some(stream.session_id);
                if let Some(thinking_content) = stream
                    .thinking_content
                    .filter(|thinking| !thinking.is_empty())
                {
                    model.push_log(LogKind::Info, "assistant reasoning", thinking_content);
                }
                if stream.content.is_empty() {
                    model.push_log(
                        LogKind::Warning,
                        "assistant",
                        "Run completed without assistant content",
                    );
                } else {
                    model.push_log(LogKind::Success, "assistant", stream.content);
                }
            }
        }
        RunStatus::Failed => {
            if let Some(stream) = model.active_stream.take() {
                model.push_log(
                    LogKind::Error,
                    "agent",
                    stream
                        .error
                        .or(event.content)
                        .unwrap_or_else(|| "Run failed".to_string()),
                );
            }
        }
        RunStatus::Cancelled => {
            model.active_stream = None;
            model.push_log(LogKind::Warning, "agent", "Run cancelled");
        }
    }

    Vec::new()
}

fn single_request(
    model: &mut Model,
    pending: PendingRequest,
    method: Method,
    params: impl serde::Serialize,
) -> Vec<Effect> {
    match enqueue_request(model, pending, method, params) {
        Ok(effect) => vec![effect],
        Err(error) => {
            model.push_log(LogKind::Error, "request", error);
            Vec::new()
        }
    }
}

fn enqueue_request(
    model: &mut Model,
    pending: PendingRequest,
    method: Method,
    params: impl serde::Serialize,
) -> Result<Effect, String> {
    let frame = Frame::request(method, params).map_err(|error| error.to_string())?;
    let request_id = match &frame {
        Frame::Request { id, .. } => id.clone(),
        _ => unreachable!(),
    };
    if matches!(pending, PendingRequest::Status { .. }) {
        model.mark_status_request();
    }
    model.pending_requests.insert(request_id, pending);
    Ok(Effect::Send(frame))
}

fn decode_response<T: serde::de::DeserializeOwned>(
    model: &mut Model,
    label: &str,
    payload: serde_json::Value,
) -> Option<T> {
    match serde_json::from_value(payload) {
        Ok(response) => Some(response),
        Err(error) => {
            model.push_log(LogKind::Error, label, format!("Invalid response: {error}"));
            None
        }
    }
}

fn format_tool_activity(tool_name: &str, tool_call_id: Option<&str>) -> String {
    match tool_call_id {
        Some(tool_call_id) => format!("{tool_name} ({tool_call_id})"),
        None => tool_name.to_string(),
    }
}

fn format_tools_catalog(tools: &[ToolEntry]) -> String {
    if tools.is_empty() {
        return "No tools available".to_string();
    }

    tools
        .iter()
        .map(|tool| {
            let availability = if tool.available {
                "available"
            } else {
                "unavailable"
            };
            match tool.spec.contract_version.as_deref() {
                Some(contract_version) => format!(
                    "{} [{}] source={} contract={}",
                    tool.spec.name, availability, tool.source, contract_version
                ),
                None => format!(
                    "{} [{}] source={}",
                    tool.spec.name, availability, tool.source
                ),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_session_list(response: &SessionListResponse) -> String {
    if response.sessions.is_empty() {
        return "No sessions".to_string();
    }

    response
        .sessions
        .iter()
        .map(|session| {
            format!(
                "{} name={} promptCollection={} messages={} createdAt={} lastActiveAt={}",
                session.session_id,
                session.name.as_deref().unwrap_or("-"),
                session.prompt_collection_id.as_deref().unwrap_or("-"),
                session.message_count,
                session.created_at,
                session.last_active_at,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_session_get(response: &SessionGetResponse) -> String {
    format!(
        "session={}\nname={}\npromptCollection={}\nmessages={}\ncreatedAt={}",
        response.session_id,
        response.name.as_deref().unwrap_or("-"),
        response.prompt_collection_id.as_deref().unwrap_or("-"),
        response.messages.len(),
        response.created_at,
    )
}

fn format_cron_list(response: &CronListResponse) -> String {
    if response.jobs.is_empty() {
        return "No cron jobs".to_string();
    }

    response
        .jobs
        .iter()
        .map(|job| {
            format!(
                "{} name={} schedule={} enabled={} lastRunAt={} nextRunAt={}",
                job.job_id,
                job.name,
                job.schedule,
                job.enabled,
                job.last_run_at.as_deref().unwrap_or("-"),
                job.next_run_at.as_deref().unwrap_or("-"),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_prompt_document_list(response: &PromptDocumentListResponse) -> String {
    if response.documents.is_empty() {
        return "No prompt documents".to_string();
    }

    response
        .documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_prompt_collection_list(response: &PromptCollectionListResponse) -> String {
    if response.collections.is_empty() {
        return "No prompt collections".to_string();
    }

    response
        .collections
        .iter()
        .map(|collection| {
            format!(
                "{} name={} documents={} description={}",
                collection.id,
                collection.name,
                collection.documents.join(","),
                collection.description.as_deref().unwrap_or("-"),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn handle_image_generate_response(model: &mut Model, response: ImageGenerateResponse) {
    if response.images.is_empty() {
        model.push_log(
            LogKind::Warning,
            "generate image",
            "Gateway returned no images",
        );
        return;
    }

    let mut saved_paths = Vec::new();
    for image in response.images {
        match save_generated_image(&model.workspace_root, &image.image_data, image.media_type.as_deref()) {
            Ok(path) => saved_paths.push(path),
            Err(error) => {
                model.push_log(
                    LogKind::Error,
                    "generate image",
                    format!("Failed to save generated image: {error}"),
                );
                return;
            }
        }
    }

    let saved_list = saved_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    model.push_log(
        LogKind::Success,
        "generate image",
        format!(
            "Saved {} image(s):\n{}\n\n[{}ms]",
            saved_paths.len(),
            saved_list,
            response.inference_time_ms
        ),
    );
}

fn handle_audio_generate_response(model: &mut Model, response: AudioGenerateResponse) {
    match save_generated_audio(
        &model.workspace_root,
        &response.audio_data,
        response.media_type.as_deref(),
        &response.format,
    ) {
        Ok(path) => model.push_log(
            LogKind::Success,
            "generate audio",
            format!(
                "Saved generated audio to {}\n\n[{}ms]",
                path.display(),
                response.inference_time_ms
            ),
        ),
        Err(error) => model.push_log(
            LogKind::Error,
            "generate audio",
            format!("Failed to save generated audio: {error}"),
        ),
    }
}

fn save_generated_image(
    workspace_root: &Path,
    base64_data: &str,
    media_type: Option<&str>,
) -> Result<PathBuf, String> {
    let extension = extension_from_image_media_type(media_type).unwrap_or("png");
    let target = generated_media_path(workspace_root, "datasets/images/generated", extension);
    let bytes = decode_base64(base64_data)?;
    save_bytes(&target, &bytes)?;
    Ok(target)
}

fn save_generated_audio(
    workspace_root: &Path,
    base64_data: &str,
    media_type: Option<&str>,
    format: &str,
) -> Result<PathBuf, String> {
    let extension = extension_from_audio_media_type(media_type)
        .or_else(|| extension_from_audio_format(format))
        .unwrap_or("wav");
    let target = generated_media_path(workspace_root, "datasets/audio/generated", extension);
    let bytes = decode_base64(base64_data)?;
    save_bytes(&target, &bytes)?;
    Ok(target)
}

fn decode_base64(data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| format!("Invalid base64 payload: {error}"))
}

fn generated_media_path(workspace_root: &Path, relative_dir: &str, extension: &str) -> PathBuf {
    let timestamp = Local::now().format("%d-%m-%y_%H-%M").to_string();
    let file_name = format!("{timestamp}.{extension}");
    unique_path(workspace_root.join(relative_dir), file_name)
}

fn unique_path(dir: PathBuf, file_name: String) -> PathBuf {
    let stem = file_name
        .rsplit_once('.')
        .map_or(file_name.as_str(), |(head, _)| head)
        .to_string();
    let ext = file_name
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_string());

    let mut candidate = dir.join(&file_name);
    let mut suffix = 1usize;
    while candidate.exists() {
        let next_name = match &ext {
            Some(ext) => format!("{stem}-{suffix}.{ext}"),
            None => format!("{stem}-{suffix}"),
        };
        candidate = dir.join(next_name);
        suffix += 1;
    }
    candidate
}

fn save_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Err("Output path has no parent directory".to_string());
    };
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("Failed to create output directory '{}': {error}", parent.display()))?;
    std::fs::write(path, bytes)
        .map_err(|error| format!("Failed to write '{}': {error}", path.display()))
}

fn extension_from_image_media_type(media_type: Option<&str>) -> Option<&'static str> {
    match media_type {
        Some("image/png") => Some("png"),
        Some("image/jpeg") => Some("jpg"),
        Some("image/webp") => Some("webp"),
        Some("image/gif") => Some("gif"),
        Some("image/bmp") => Some("bmp"),
        Some("image/tiff") => Some("tiff"),
        Some("image/avif") => Some("avif"),
        Some("image/heic") => Some("heic"),
        _ => None,
    }
}

fn extension_from_audio_media_type(media_type: Option<&str>) -> Option<&'static str> {
    match media_type {
        Some("audio/wav") => Some("wav"),
        Some("audio/mpeg") => Some("mp3"),
        Some("audio/flac") => Some("flac"),
        Some("audio/ogg") => Some("ogg"),
        Some("audio/opus") => Some("opus"),
        Some("audio/mp4") => Some("m4a"),
        Some("audio/aac") => Some("aac"),
        Some("audio/webm") => Some("webm"),
        Some("audio/L16") => Some("pcm"),
        _ => None,
    }
}

fn extension_from_audio_format(format: &str) -> Option<&'static str> {
    match format {
        "wav" => Some("wav"),
        "mp3" => Some("mp3"),
        "pcm" => Some("pcm"),
        _ => None,
    }
}

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}
