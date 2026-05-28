use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use nexo_ws_schema::{HelloOk, RunStatus};
use ratatui::layout::Rect;

use super::completion;

const MAX_LOG_ENTRIES: usize = 250;
const STATUS_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const RECONNECT_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityButton {
    CopyAll,
    CopyLastOutput,
}

#[derive(Debug, Clone, Default)]
pub struct StartOptions {
    pub url_override: Option<String>,
    pub initial_session_id: Option<String>,
    pub initial_session_name: Option<String>,
    pub initial_model_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunningState {
    #[default]
    Running,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogKind {
    Info,
    Success,
    Warning,
    Error,
    Command,
    Event,
    Response,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub kind: LogKind,
    pub title: String,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub label: String,
    pub replacement: String,
}

#[derive(Debug, Clone)]
pub struct CompletionState {
    pub items: Vec<CompletionItem>,
    pub selected: usize,
    pub range: (usize, usize),
}

#[derive(Debug, Clone)]
pub enum PendingRequest {
    Health,
    Status { silent: bool },
    Send,
    RunStart,
    SessionCreate,
    SessionList,
    SessionGet,
    SessionClear,
    ToolsCatalog,
    ToolsExecute,
    CronCreate,
    CronList,
    CronDelete,
    PromptDocumentCreate,
    PromptDocumentList,
    PromptDocumentDelete,
    PromptCollectionCreate,
    PromptCollectionList,
    PromptCollectionDelete,
    SystemPresence,
    ImageAnalyze,
}

impl PendingRequest {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::Status { .. } => "status",
            Self::Send => "send",
            Self::RunStart => "run",
            Self::SessionCreate => "session create",
            Self::SessionList => "session list",
            Self::SessionGet => "session get",
            Self::SessionClear => "session clear",
            Self::ToolsCatalog => "tools catalog",
            Self::ToolsExecute => "tools execute",
            Self::CronCreate => "cron create",
            Self::CronList => "cron list",
            Self::CronDelete => "cron delete",
            Self::PromptDocumentCreate => "prompt document create",
            Self::PromptDocumentList => "prompt document list",
            Self::PromptDocumentDelete => "prompt document delete",
            Self::PromptCollectionCreate => "prompt collection create",
            Self::PromptCollectionList => "prompt collection list",
            Self::PromptCollectionDelete => "prompt collection delete",
            Self::SystemPresence => "system presence",
            Self::ImageAnalyze => "image analyze",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SummaryStatus {
    pub connected: bool,
    pub gateway_url: String,
    pub protocol: Option<u32>,
    pub connected_nodes: Option<u32>,
    pub connected_clients: Option<u32>,
    pub capabilities: Vec<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ActiveStream {
    pub run_id: String,
    pub session_id: String,
    pub status: RunStatus,
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub thinking_content: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub gateway_url: String,
    pub hello: HelloOk,
}

#[derive(Debug)]
pub struct Model {
    pub running_state: RunningState,
    pub input: String,
    pub cursor: usize,
    pub logs: VecDeque<LogEntry>,
    pub activity_scroll: u16,
    pub activity_area: Rect,
    pub activity_copy_all_button: Rect,
    pub activity_copy_last_output_button: Rect,
    pub activity_total_lines: usize,
    pub activity_viewport_lines: usize,
    pub completion: Option<CompletionState>,
    pub show_help: bool,
    pub summary: SummaryStatus,
    pub pending_requests: HashMap<String, PendingRequest>,
    pub active_stream: Option<ActiveStream>,
    pub workspace_root: PathBuf,
    pub current_session_id: Option<String>,
    pub default_session_name: Option<String>,
    pub default_model_id: Option<String>,
    pub startup_session_name: Option<String>,
    pub last_status_request_at: Option<Instant>,
    pub last_reconnect_attempt_at: Option<Instant>,
}

impl Model {
    pub fn new(connection: ConnectionInfo, options: StartOptions, workspace_root: PathBuf) -> Self {
        let mut model = Self {
            running_state: RunningState::Running,
            input: String::new(),
            cursor: 0,
            logs: VecDeque::new(),
            activity_scroll: 0,
            activity_area: Rect::default(),
            activity_copy_all_button: Rect::default(),
            activity_copy_last_output_button: Rect::default(),
            activity_total_lines: 0,
            activity_viewport_lines: 0,
            completion: None,
            show_help: false,
            summary: SummaryStatus {
                connected: true,
                gateway_url: connection.gateway_url.clone(),
                protocol: Some(connection.hello.protocol),
                connected_nodes: None,
                connected_clients: None,
                capabilities: Vec::new(),
                last_error: None,
            },
            pending_requests: HashMap::new(),
            active_stream: None,
            workspace_root,
            current_session_id: options.initial_session_id.clone(),
            default_session_name: options.initial_session_name.clone(),
            default_model_id: options.initial_model_id,
            startup_session_name: options
                .initial_session_id
                .is_none()
                .then_some(options.initial_session_name)
                .flatten(),
            last_status_request_at: None,
            last_reconnect_attempt_at: None,
        };

        model.push_log(
            LogKind::Success,
            "Connected",
            format!(
                "Connected to {} with protocol v{}",
                connection.gateway_url, connection.hello.protocol
            ),
        );
        if let Some(session_id) = &model.current_session_id {
            model.push_log(
                LogKind::Info,
                "Session",
                format!("Using initial session {session_id}"),
            );
        }
        if let Some(model_id) = &model.default_model_id {
            model.push_log(
                LogKind::Info,
                "Model",
                format!("Using default model {model_id}"),
            );
        }
        model.push_log(
            LogKind::Info,
            "Welcome",
            "Type /help to see commands. Use Tab to autocomplete commands and @file references.",
        );
        model
    }

    pub fn push_log(&mut self, kind: LogKind, title: impl Into<String>, body: impl Into<String>) {
        self.logs.push_back(LogEntry {
            kind,
            title: title.into(),
            body: body.into(),
        });
        while self.logs.len() > MAX_LOG_ENTRIES {
            self.logs.pop_front();
        }
    }

    pub fn clear_logs(&mut self) {
        self.logs.clear();
        self.active_stream = None;
        self.activity_scroll = 0;
        self.activity_total_lines = 0;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.refresh_completion();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let previous = self.input[..self.cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.input.replace_range(previous..self.cursor, "");
        self.cursor = previous;
        self.refresh_completion();
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }

        let next = self.input[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.cursor + index)
            .unwrap_or(self.input.len());
        self.input.replace_range(self.cursor..next, "");
        self.refresh_completion();
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }

        self.cursor = self.input[..self.cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.refresh_completion();
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }

        self.cursor = self.input[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.cursor + index)
            .unwrap_or(self.input.len());
        self.refresh_completion();
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor = 0;
        self.refresh_completion();
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor = self.input.len();
        self.refresh_completion();
    }

    pub fn accept_completion(&mut self) {
        let Some(completion) = self.completion.clone() else {
            return;
        };
        let Some(item) = completion.items.get(completion.selected) else {
            return;
        };

        self.input
            .replace_range(completion.range.0..completion.range.1, &item.replacement);
        self.cursor = completion.range.0 + item.replacement.len();
        self.refresh_completion();
    }

    pub fn select_next_completion(&mut self) {
        if let Some(completion) = &mut self.completion
            && !completion.items.is_empty()
        {
            completion.selected = (completion.selected + 1) % completion.items.len();
        }
    }

    pub fn select_prev_completion(&mut self) {
        if let Some(completion) = &mut self.completion
            && !completion.items.is_empty()
        {
            completion.selected = if completion.selected == 0 {
                completion.items.len() - 1
            } else {
                completion.selected - 1
            };
        }
    }

    pub fn clear_completion(&mut self) {
        self.completion = None;
    }

    pub fn take_input(&mut self) -> String {
        self.cursor = 0;
        self.completion = None;
        std::mem::take(&mut self.input)
    }

    pub fn refresh_completion(&mut self) {
        self.completion = completion::compute(&self.workspace_root, &self.input, self.cursor);
    }

    pub fn mark_status_request(&mut self) {
        self.last_status_request_at = Some(Instant::now());
    }

    pub fn should_refresh_status(&self) -> bool {
        self.summary.connected
            && self
                .last_status_request_at
                .is_none_or(|last| last.elapsed() >= STATUS_REFRESH_INTERVAL)
    }

    pub fn should_retry_connect(&self) -> bool {
        !self.summary.connected
            && self
                .last_reconnect_attempt_at
                .is_none_or(|last| last.elapsed() >= RECONNECT_INTERVAL)
    }

    pub fn mark_reconnect_attempt(&mut self) {
        self.last_reconnect_attempt_at = Some(Instant::now());
    }

    pub fn set_connected(&mut self, connection: ConnectionInfo) {
        self.summary.connected = true;
        self.summary.gateway_url = connection.gateway_url.clone();
        self.summary.protocol = Some(connection.hello.protocol);
        self.summary.last_error = None;
        self.last_status_request_at = None;
        self.last_reconnect_attempt_at = None;
        self.pending_requests.clear();
        self.push_log(
            LogKind::Success,
            "Reconnected",
            format!(
                "Connected to {} with protocol v{}",
                connection.gateway_url, connection.hello.protocol
            ),
        );
    }

    pub fn set_disconnected(&mut self, error: impl Into<String>) {
        let error = error.into();
        let was_connected = self.summary.connected;
        self.summary.connected = false;
        self.summary.last_error = Some(error.clone());
        self.last_status_request_at = None;
        self.last_reconnect_attempt_at = None;
        self.pending_requests.clear();
        if was_connected {
            self.push_log(LogKind::Error, "Disconnected", error);
        }
    }

    pub fn update_activity_view(
        &mut self,
        scroll_area: Rect,
        viewport_lines: usize,
        total_lines: usize,
        copy_all_button: Rect,
        copy_last_output_button: Rect,
    ) {
        self.activity_area = scroll_area;
        self.activity_copy_all_button = copy_all_button;
        self.activity_copy_last_output_button = copy_last_output_button;
        self.activity_viewport_lines = viewport_lines;
        self.activity_total_lines = total_lines;
        self.activity_scroll = self.activity_scroll.min(self.max_activity_scroll());
    }

    pub fn scroll_activity_up(&mut self, column: u16, row: u16, amount: u16) {
        if !self.activity_contains(column, row) {
            return;
        }

        self.activity_scroll = self
            .activity_scroll
            .saturating_add(amount)
            .min(self.max_activity_scroll());
    }

    pub fn scroll_activity_down(&mut self, column: u16, row: u16, amount: u16) {
        if !self.activity_contains(column, row) {
            return;
        }

        self.activity_scroll = self.activity_scroll.saturating_sub(amount);
    }

    #[cfg(test)]
    pub fn activity_start_line(&self) -> usize {
        self.activity_total_lines
            .saturating_sub(self.activity_viewport_lines + self.activity_scroll as usize)
    }

    pub fn activity_button_at(&self, column: u16, row: u16) -> Option<ActivityButton> {
        if rect_contains(self.activity_copy_all_button, column, row) {
            Some(ActivityButton::CopyAll)
        } else if rect_contains(self.activity_copy_last_output_button, column, row) {
            Some(ActivityButton::CopyLastOutput)
        } else {
            None
        }
    }

    pub fn all_activity_text(&self) -> String {
        self.activity_entries()
            .into_iter()
            .filter(|entry| !is_clipboard_activity(entry))
            .map(|entry| format_activity_entry(&entry))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn last_output_text(&self) -> Option<String> {
        if let Some(stream) = &self.active_stream {
            if !stream.content.is_empty() {
                return Some(stream.content.clone());
            }
            if let Some(thinking_content) = &stream.thinking_content
                && !thinking_content.is_empty()
            {
                return Some(thinking_content.clone());
            }
            if let Some(error) = &stream.error {
                return Some(error.clone());
            }
        }

        self.logs
            .iter()
            .rev()
            .find(|entry| entry.kind != LogKind::Command && !is_clipboard_activity(entry))
            .map(|entry| entry.body.clone())
    }

    fn max_activity_scroll(&self) -> u16 {
        self.activity_total_lines
            .saturating_sub(self.activity_viewport_lines)
            .min(u16::MAX as usize) as u16
    }

    fn activity_contains(&self, column: u16, row: u16) -> bool {
        let x_end = self
            .activity_area
            .x
            .saturating_add(self.activity_area.width);
        let y_end = self
            .activity_area
            .y
            .saturating_add(self.activity_area.height);
        column >= self.activity_area.x
            && column < x_end
            && row >= self.activity_area.y
            && row < y_end
    }

    fn activity_entries(&self) -> Vec<LogEntry> {
        let mut entries: Vec<LogEntry> = self.logs.iter().cloned().collect();
        if let Some(stream) = &self.active_stream {
            let title = match (&stream.tool_name, &stream.tool_call_id) {
                (Some(tool_name), Some(tool_call_id)) => format!(
                    "assistant ({:?}, tool: {tool_name}, call: {tool_call_id})",
                    stream.status
                ),
                (Some(tool_name), None) => {
                    format!("assistant ({:?}, tool: {tool_name})", stream.status)
                }
                _ => format!("assistant ({:?})", stream.status),
            };
            entries.push(LogEntry {
                kind: LogKind::Response,
                title,
                body: if stream.content.is_empty() {
                    stream
                        .thinking_content
                        .clone()
                        .or_else(|| stream.error.clone())
                        .unwrap_or_else(|| "...waiting for response...".to_string())
                } else {
                    stream.content.clone()
                },
            });
        }
        entries
    }
}

fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
    let x_end = rect.x.saturating_add(rect.width);
    let y_end = rect.y.saturating_add(rect.height);
    rect.width > 0
        && rect.height > 0
        && column >= rect.x
        && column < x_end
        && row >= rect.y
        && row < y_end
}

fn format_activity_entry(entry: &LogEntry) -> String {
    format!("[{}]\n{}", entry.title, entry.body)
}

fn is_clipboard_activity(entry: &LogEntry) -> bool {
    entry.title == "clipboard"
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn make_model() -> Model {
        Model::new(
            ConnectionInfo {
                gateway_url: "ws://127.0.0.1:6969".into(),
                hello: HelloOk::default(),
            },
            StartOptions::default(),
            std::env::current_dir().unwrap(),
        )
    }

    #[test]
    fn activity_scroll_clamps_to_available_history() {
        let mut model = make_model();
        model.update_activity_view(
            Rect {
                x: 0,
                y: 4,
                width: 80,
                height: 10,
            },
            8,
            30,
            Rect::default(),
            Rect::default(),
        );

        model.scroll_activity_up(10, 5, 100);
        assert_eq!(model.activity_scroll, 22);
        assert_eq!(model.activity_start_line(), 0);
    }

    #[test]
    fn activity_scroll_ignores_mouse_outside_panel() {
        let mut model = make_model();
        model.update_activity_view(
            Rect {
                x: 0,
                y: 4,
                width: 80,
                height: 10,
            },
            8,
            30,
            Rect::default(),
            Rect::default(),
        );

        model.scroll_activity_up(100, 100, 1);
        assert_eq!(model.activity_scroll, 0);
    }

    #[test]
    fn activity_button_hit_detection_returns_copy_all() {
        let mut model = make_model();
        model.update_activity_view(
            Rect {
                x: 0,
                y: 4,
                width: 80,
                height: 10,
            },
            8,
            30,
            Rect {
                x: 5,
                y: 4,
                width: 10,
                height: 1,
            },
            Rect {
                x: 16,
                y: 4,
                width: 18,
                height: 1,
            },
        );

        assert_eq!(
            model.activity_button_at(6, 4),
            Some(ActivityButton::CopyAll)
        );
        assert_eq!(
            model.activity_button_at(20, 4),
            Some(ActivityButton::CopyLastOutput)
        );
    }

    #[test]
    fn last_output_text_skips_command_entries() {
        let mut model = make_model();
        model.push_log(LogKind::Command, "Input", "/status");
        model.push_log(LogKind::Response, "status", "connected=true");

        assert_eq!(model.last_output_text().as_deref(), Some("connected=true"));
    }

    #[test]
    fn last_output_text_ignores_clipboard_feedback() {
        let mut model = make_model();
        model.push_log(LogKind::Response, "assistant", "real output");
        model.push_log(
            LogKind::Success,
            "clipboard",
            "Copied last output to clipboard",
        );

        assert_eq!(model.last_output_text().as_deref(), Some("real output"));
    }

    #[test]
    fn last_output_text_uses_reasoning_for_active_stream() {
        let mut model = make_model();
        model.active_stream = Some(ActiveStream {
            run_id: "run-1".into(),
            session_id: "session-1".into(),
            status: RunStatus::Thinking,
            content: String::new(),
            tool_name: None,
            tool_call_id: None,
            thinking_content: Some("planning next step".into()),
            error: None,
        });

        assert_eq!(
            model.last_output_text().as_deref(),
            Some("planning next step")
        );
    }

    #[test]
    fn all_activity_text_ignores_clipboard_feedback() {
        let mut model = make_model();
        model.clear_logs();
        model.push_log(LogKind::Info, "Session", "session ready");
        model.push_log(
            LogKind::Success,
            "clipboard",
            "Copied all activity to clipboard",
        );
        model.push_log(LogKind::Response, "assistant", "real output");

        assert_eq!(
            model.all_activity_text(),
            "[Session]\nsession ready\n\n[assistant]\nreal output"
        );
    }
}
