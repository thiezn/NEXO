use super::{NexoUserState, TuiAction};
use crate::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use nexo_core::SessionId;
use std::time::Duration;

const COMMAND_SUGGESTIONS: &[&str] = &[
    "/state",
    "/sessions",
    "/session get ",
    "/session clear ",
    "/cancel ",
    "/disconnect",
    "/quit",
];

/// Parsed terminal events consumed by the TUI controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiInputEvent {
    /// Insert a printable character into the prompt.
    InsertChar(char),
    /// Delete the previous character.
    Backspace,
    /// Move the prompt cursor left.
    MoveLeft,
    /// Move the prompt cursor right.
    MoveRight,
    /// Scroll the history pane up.
    ScrollUp,
    /// Scroll the history pane down.
    ScrollDown,
    /// Submit the current prompt.
    Submit,
    /// Accept the currently selected autocomplete suggestion.
    AcceptAutocomplete,
    /// Select the next autocomplete suggestion.
    SelectNextAutocomplete,
    /// Select the previous autocomplete suggestion.
    SelectPreviousAutocomplete,
    /// Dismiss the autocomplete popup.
    DismissAutocomplete,
    /// Request full application shutdown.
    RequestShutdown,
    /// No state-changing action is required.
    Noop,
}

/// Result of submitting the current prompt.
#[derive(Debug, Clone)]
pub enum InputCommand {
    /// Dispatch a TUI action to the engine.
    Action {
        /// The action to send to the engine.
        action: TuiAction,

        /// The raw user-entered text that produced the action.
        submitted_text: String,

        /// Whether the input was entered in slash-command mode.
        is_command: bool,
    },
    /// Surface a local controller message without sending anything to the engine.
    LocalMessage(String),
}

/// Local prompt and viewport state for the terminal UI.
#[derive(Debug, Clone, Default)]
pub struct InputState {
    buffer: String,
    cursor: usize,
    history_scroll: usize,
    autocomplete_items: Vec<String>,
    autocomplete_selected: usize,
}

impl InputState {
    /// Creates a new empty prompt state.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current prompt buffer.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Returns the current prompt cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Returns the current history scroll offset.
    pub fn history_scroll(&self) -> usize {
        self.history_scroll
    }

    /// Returns the current autocomplete suggestions.
    pub fn autocomplete_items(&self) -> &[String] {
        &self.autocomplete_items
    }

    /// Returns the selected autocomplete index, if any.
    pub fn autocomplete_selected(&self) -> Option<usize> {
        (!self.autocomplete_items.is_empty()).then_some(self.autocomplete_selected)
    }

    /// Inserts a character at the current cursor position.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character to insert into the prompt buffer.
    pub fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    /// Deletes the character before the current cursor position.
    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }

        self.cursor -= 1;
        self.buffer.remove(self.cursor);
    }

    /// Moves the prompt cursor left by one character.
    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Moves the prompt cursor right by one character.
    pub fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.buffer.len());
    }

    /// Scrolls the history view upward.
    pub fn scroll_up(&mut self) {
        self.history_scroll = self.history_scroll.saturating_add(1);
    }

    /// Scrolls the history view downward.
    ///
    /// # Arguments
    ///
    /// * `state` - The current user state used to clamp scroll offset against available history.
    pub fn scroll_down(&mut self, state: &NexoUserState) {
        let max_index = state.timeline().len().saturating_sub(1);
        self.history_scroll = self.history_scroll.saturating_sub(1).min(max_index);
    }

    /// Recomputes autocomplete suggestions for the current prompt buffer.
    pub fn refresh_autocomplete(&mut self) {
        if !self.buffer.starts_with('/') {
            self.autocomplete_items.clear();
            self.autocomplete_selected = 0;
            return;
        }

        self.autocomplete_items = COMMAND_SUGGESTIONS
            .iter()
            .filter(|item| item.starts_with(&self.buffer))
            .map(|item| (*item).to_string())
            .collect();

        if self.autocomplete_items.is_empty() {
            self.autocomplete_selected = 0;
        } else {
            self.autocomplete_selected = self
                .autocomplete_selected
                .min(self.autocomplete_items.len().saturating_sub(1));
        }
    }

    /// Accepts the currently selected autocomplete suggestion.
    pub fn accept_autocomplete(&mut self) {
        if let Some(item) = self.autocomplete_items.get(self.autocomplete_selected) {
            self.buffer = item.clone();
            self.cursor = self.buffer.len();
        }
    }

    /// Selects the next autocomplete suggestion.
    pub fn select_next_autocomplete(&mut self) {
        if self.autocomplete_items.is_empty() {
            return;
        }

        self.autocomplete_selected = (self.autocomplete_selected + 1) % self.autocomplete_items.len();
    }

    /// Selects the previous autocomplete suggestion.
    pub fn select_previous_autocomplete(&mut self) {
        if self.autocomplete_items.is_empty() {
            return;
        }

        self.autocomplete_selected = if self.autocomplete_selected == 0 {
            self.autocomplete_items.len() - 1
        } else {
            self.autocomplete_selected - 1
        };
    }

    /// Dismisses the autocomplete popup.
    pub fn dismiss_autocomplete(&mut self) {
        self.autocomplete_items.clear();
        self.autocomplete_selected = 0;
    }

    /// Converts the current buffer into an engine command and clears the prompt.
    ///
    /// # Arguments
    ///
    /// * `state` - The current `NexoUserState`, used for session-aware prompt submission.
    pub fn submit(&mut self, state: &NexoUserState) -> Result<Option<InputCommand>> {
        let input = self.buffer.trim().to_string();
        self.buffer.clear();
        self.cursor = 0;
        self.dismiss_autocomplete();

        if input.is_empty() {
            return Ok(None);
        }

        if input.starts_with('/') {
            return Self::parse_command(input, state).map(Some);
        }

        let action = TuiAction::from_plain_text_prompt(
            input.clone(),
            state.selected_session_id().cloned(),
        )?;
        Ok(Some(InputCommand::Action {
            action,
            submitted_text: input,
            is_command: false,
        }))
    }

    /// Parses a slash command into a controller command.
    ///
    /// # Arguments
    ///
    /// * `input` - The raw slash-prefixed prompt input.
    /// * `state` - The current state used for session-aware commands.
    fn parse_command(input: String, _state: &NexoUserState) -> Result<InputCommand> {
        let trimmed = input.trim();
        if trimmed == "/state" {
            return Ok(InputCommand::Action {
                action: TuiAction::RefreshState,
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if trimmed == "/sessions" {
            return Ok(InputCommand::Action {
                action: TuiAction::ListSessions,
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if trimmed == "/disconnect" {
            return Ok(InputCommand::Action {
                action: TuiAction::Disconnect,
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if trimmed == "/quit" {
            return Ok(InputCommand::Action {
                action: TuiAction::Shutdown,
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if let Some(rest) = trimmed.strip_prefix("/session get ") {
            return Ok(InputCommand::Action {
                action: TuiAction::GetSession {
                    session_id: SessionId::from(rest.trim()),
                },
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if let Some(rest) = trimmed.strip_prefix("/session clear ") {
            return Ok(InputCommand::Action {
                action: TuiAction::ClearSession {
                    session_id: SessionId::from(rest.trim()),
                },
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        if let Some(rest) = trimmed.strip_prefix("/cancel ") {
            return Ok(InputCommand::Action {
                action: TuiAction::Cancel {
                    operation_id: nexo_core::OperationId::from(rest.trim()),
                },
                submitted_text: trimmed.into(),
                is_command: true,
            });
        }

        Ok(InputCommand::LocalMessage(format!(
            "command not implemented yet: {trimmed}"
        )))
    }
}

/// Polls for the next terminal input event without enabling mouse capture.
///
/// # Arguments
///
/// * `timeout` - The duration to wait for terminal input before returning `Ok(None)`.
pub fn poll_event(timeout: Duration) -> Result<Option<TuiInputEvent>> {
    if !event::poll(timeout)? {
        return Ok(None);
    }

    let input = match event::read()? {
        Event::Key(key) if key.kind == KeyEventKind::Press => map_key_event(key),
        Event::Resize(_, _) => TuiInputEvent::Noop,
        _ => TuiInputEvent::Noop,
    };

    Ok(Some(input))
}

/// Maps a key press into a normalized TUI input event.
///
/// # Arguments
///
/// * `key` - The raw crossterm key event.
fn map_key_event(key: KeyEvent) -> TuiInputEvent {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            TuiInputEvent::RequestShutdown
        }
        (KeyCode::Enter, _) => TuiInputEvent::Submit,
        (KeyCode::Backspace, _) => TuiInputEvent::Backspace,
        (KeyCode::Left, _) => TuiInputEvent::MoveLeft,
        (KeyCode::Right, _) => TuiInputEvent::MoveRight,
        (KeyCode::Up, _) => TuiInputEvent::ScrollUp,
        (KeyCode::Down, _) => TuiInputEvent::ScrollDown,
        (KeyCode::Tab, _) => TuiInputEvent::AcceptAutocomplete,
        (KeyCode::Esc, _) => TuiInputEvent::DismissAutocomplete,
        (KeyCode::Char(ch), modifiers)
            if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            TuiInputEvent::InsertChar(ch)
        }
        _ => TuiInputEvent::Noop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_state_command_maps_to_refresh_state() {
        let mut input = InputState::new();
        input.buffer = "/state".into();
        input.cursor = input.buffer.len();

        let command = input
            .submit(&NexoUserState::new())
            .expect("submit should parse slash command")
            .expect("submit should produce a command");

        assert!(matches!(command, InputCommand::Action { action: TuiAction::RefreshState, is_command: true, .. }));
    }
}
