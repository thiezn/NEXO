use nexo_ws_schema::{
    CronPayload, EventKind, Frame, HealthResponse, ImageAnalyzeResponse, MessagePayload, Method,
    PresencePayload, RunEventPayload, RunStartResponse, RunStatus, SendResponse,
    SessionCreateResponse, ShutdownPayload, StatusResponse,
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
            serde_json::json!({}),
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
            serde_json::json!({}),
        ),
        AppCommand::Status => single_request(
            model,
            PendingRequest::Status { silent: false },
            Method::Status,
            serde_json::json!({}),
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
            serde_json::json!({}),
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
            serde_json::json!({}),
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
            serde_json::json!({}),
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
            serde_json::json!({}),
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
            serde_json::json!({}),
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
            let response: HealthResponse = match serde_json::from_value(payload) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(
                        LogKind::Error,
                        "health",
                        format!("Invalid response: {error}"),
                    );
                    return Vec::new();
                }
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
            let response: StatusResponse = match serde_json::from_value(payload.clone()) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(
                        LogKind::Error,
                        "status",
                        format!("Invalid response: {error}"),
                    );
                    return Vec::new();
                }
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
            let response: SendResponse = match serde_json::from_value(payload) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(LogKind::Error, "send", format!("Invalid response: {error}"));
                    return Vec::new();
                }
            };
            model.push_log(
                LogKind::Success,
                "send",
                format!("delivered={}", response.delivered),
            );
        }
        PendingRequest::RunStart => {
            let response: RunStartResponse = match serde_json::from_value(payload) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(LogKind::Error, "run", format!("Invalid response: {error}"));
                    return Vec::new();
                }
            };
            model.current_session_id = Some(response.session_id.clone());
            model.active_stream = Some(ActiveStream {
                run_id: response.run_id.clone(),
                session_id: response.session_id.clone(),
                status: response.status,
                content: String::new(),
                tool_name: None,
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
        }
        PendingRequest::SessionCreate => {
            let response: SessionCreateResponse = match serde_json::from_value(payload) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(
                        LogKind::Error,
                        "session create",
                        format!("Invalid response: {error}"),
                    );
                    return Vec::new();
                }
            };
            model.current_session_id = Some(response.session_id.clone());
            model.push_log(
                LogKind::Success,
                "session create",
                format!("Current session is {}", response.session_id),
            );
        }
        PendingRequest::ImageAnalyze => {
            let response: ImageAnalyzeResponse = match serde_json::from_value(payload) {
                Ok(response) => response,
                Err(error) => {
                    model.push_log(
                        LogKind::Error,
                        "image analyze",
                        format!("Invalid response: {error}"),
                    );
                    return Vec::new();
                }
            };
            model.push_log(
                LogKind::Success,
                "image analyze",
                format!(
                    "{}\n\n[{} tokens in {}ms]",
                    response.text, response.tokens_generated, response.inference_time_ms
                ),
            );
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
    stream.error = event.error.clone();
    if matches!(event.status, RunStatus::Streaming | RunStatus::Accepted)
        && let Some(content) = &event.content
    {
        stream.content = content.clone();
    }

    match event.status {
        RunStatus::Queued => model.push_log(
            LogKind::Info,
            "agent",
            "Run queued, waiting for an inference node",
        ),
        RunStatus::Thinking => {}
        RunStatus::ToolCall => {
            let tool_name = event.tool_name.unwrap_or_else(|| "unknown".to_string());
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
                model.push_log(LogKind::Success, "assistant", stream.content);
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

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}
