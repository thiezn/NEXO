use super::{InputCommand, InputState, NexoUserState, TuiAction, TuiEvent, input, renderer};
use crate::Result;
use crossterm::ExecutableCommand;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{self, stdout};
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::time;

/// Controller for terminal user interface interactions in the `nexo-user` application.
pub struct TuiController {
    state: NexoUserState,
    input_state: InputState,
}

impl TuiController {
    /// Creates a new instance of the TUI controller.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn new() -> Self {
        Self {
            state: NexoUserState::new(),
            input_state: InputState::new(),
        }
    }

    /// Applies a UI event to the controller state.
    ///
    /// # Arguments
    ///
    /// * `event` - The event emitted by the engine that should update the visible UI state.
    pub fn apply_event(&mut self, event: &TuiEvent) {
        self.state.apply_event(event);
    }

    /// Returns the current application state rendered by the TUI.
    pub fn state(&self) -> &NexoUserState {
        &self.state
    }

    /// Runs the controller event loop over engine channels.
    ///
    /// # Arguments
    ///
    /// * `action_tx` - The sender used to submit user actions to the engine.
    /// * `event_rx` - The receiver used to consume engine events for UI updates.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the controller loop exits normally.
    pub async fn run(
        &mut self,
        action_tx: Sender<TuiAction>,
        event_rx: &mut Receiver<TuiEvent>,
    ) -> Result {
        let mut terminal = TerminalSession::new()?;
        let mut tick = time::interval(Duration::from_millis(100));

        loop {
            terminal.draw(|frame| {
                renderer::render(frame, &self.state, &self.input_state);
            })?;

            tokio::select! {
                Some(event) = event_rx.recv() => {
                    self.apply_event(&event);

                    if matches!(event, TuiEvent::Disconnected | TuiEvent::ShutdownRequested) {
                        break;
                    }
                }
                _ = tick.tick() => {
                    if let Some(event) = input::poll_event(Duration::from_millis(1))? {
                        if !self.handle_input_event(event, &action_tx).await? {
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handles a parsed terminal input event.
    ///
    /// # Arguments
    ///
    /// * `event` - The parsed input event emitted by the input module.
    /// * `action_tx` - The sender used to dispatch resulting TUI actions to the engine.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` to continue the loop or `Ok(false)` to terminate it.
    async fn handle_input_event(
        &mut self,
        event: super::TuiInputEvent,
        action_tx: &Sender<TuiAction>,
    ) -> Result<bool> {
        match event {
            super::TuiInputEvent::InsertChar(ch) => self.input_state.insert_char(ch),
            super::TuiInputEvent::Backspace => self.input_state.backspace(),
            super::TuiInputEvent::MoveLeft => self.input_state.move_left(),
            super::TuiInputEvent::MoveRight => self.input_state.move_right(),
            super::TuiInputEvent::ScrollUp => self.input_state.scroll_up(),
            super::TuiInputEvent::ScrollDown => self.input_state.scroll_down(&self.state),
            super::TuiInputEvent::AcceptAutocomplete => self.input_state.accept_autocomplete(),
            super::TuiInputEvent::SelectNextAutocomplete => {
                self.input_state.select_next_autocomplete()
            }
            super::TuiInputEvent::SelectPreviousAutocomplete => {
                self.input_state.select_previous_autocomplete()
            }
            super::TuiInputEvent::DismissAutocomplete => self.input_state.dismiss_autocomplete(),
            super::TuiInputEvent::Submit => {
                if let Some(command) = self.input_state.submit(&self.state)? {
                    match command {
                        InputCommand::Action {
                            action,
                            submitted_text,
                            is_command,
                        } => {
                            if is_command {
                                self.state.record_user_command(&submitted_text);
                            } else {
                                self.state.record_user_prompt(&submitted_text);
                            }
                            action_tx.send(action).await?;
                        }
                        InputCommand::LocalMessage(message) => {
                            self.state.apply_event(&TuiEvent::Error {
                                context: "command".into(),
                                message,
                            });
                        }
                    }
                }
            }
            super::TuiInputEvent::RequestShutdown => {
                action_tx.send(TuiAction::Shutdown).await?;
                return Ok(false);
            }
            super::TuiInputEvent::Noop => {}
        }

        self.input_state.refresh_autocomplete();
        Ok(true)
    }
}

/// Terminal lifecycle wrapper for the ratatui session.
struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalSession {
    /// Creates a new terminal session in raw alternate-screen mode.
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut out = stdout();
        out.execute(EnterAlternateScreen)?;

        let terminal = Terminal::new(CrosstermBackend::new(stdout()))
            .map_err(|error| crate::Error::Other(format!("failed to create terminal: {error}")))?;

        Ok(Self { terminal })
    }

    /// Draws one ratatui frame.
    ///
    /// # Arguments
    ///
    /// * `draw_fn` - The closure that renders the current frame.
    fn draw(&mut self, draw_fn: impl FnOnce(&mut ratatui::Frame<'_>)) -> Result {
        self.terminal
            .draw(draw_fn)
            .map(|_| ())
            .map_err(|error| crate::Error::Other(format!("terminal draw failed: {error}")))
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = stdout().execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}
