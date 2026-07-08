use super::TuiEvent;
use nexo_core::{NexoState, OperationId, Session, SessionId, Sessions};
use std::collections::HashMap;

/// The actor responsible for a timeline entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistorySource {
    /// An action submitted by the local user.
    User,

    /// A control or state message received from the gateway.
    Gateway,

    /// An inference response emitted by the model runtime.
    Inference,

    /// A tool-related activity message.
    Tool,

    /// A terminal or application-local error.
    Error,
}

/// The semantic category of a timeline entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// A freeform user prompt.
    UserPrompt,

    /// A slash command submitted by the user.
    UserCommand,

    /// A gateway control or lifecycle message.
    GatewayControl,

    /// A state update emitted by the gateway.
    GatewayState,

    /// A generic operation acknowledgement or completion.
    Operation,

    /// A plain text model output chunk.
    InferenceText,

    /// A model thinking or reasoning chunk.
    InferenceThinking,

    /// A tool call or tool result activity.
    ToolActivity,

    /// An error message.
    Error,
}

/// A structured item displayed in the history pane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    /// The actor responsible for the entry.
    pub source: HistorySource,

    /// The semantic category of the entry.
    pub kind: HistoryKind,

    /// The user-visible body content of the entry.
    pub body: String,
}

/// The connection lifecycle visible to the terminal UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionStatus {
    /// The user is not connected to the gateway.
    #[default]
    Disconnected,

    /// The user is currently connected to the gateway.
    Connected,
}

/// The engine-owned state model rendered by the terminal UI.
#[derive(Debug, Clone, Default)]
pub struct NexoUserState {
    connection_status: ConnectionStatus,
    state: Option<NexoState>,
    sessions: Sessions,
    selected_session_id: Option<SessionId>,
    active_operations: HashMap<OperationId, String>,
    timeline: Vec<HistoryEntry>,
}

impl NexoUserState {
    /// Creates a new, empty application state.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a UI event to the stored application state.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to fold into the persistent user-facing state.
    pub fn apply_event(&mut self, event: &TuiEvent) {
        match event {
            TuiEvent::Connected => {
                self.connection_status = ConnectionStatus::Connected;
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayControl,
                    "gateway connection established",
                );
            }
            TuiEvent::Disconnected => {
                self.connection_status = ConnectionStatus::Disconnected;
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayControl,
                    "gateway connection closed",
                );
                self.active_operations.clear();
            }
            TuiEvent::ShutdownRequested => {
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayControl,
                    "shutdown requested",
                );
            }
            TuiEvent::StateUpdated { state } => {
                self.state = Some(state.clone());
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayState,
                    &format!(
                        "gateway state updated (users: {}, nodes: {})",
                        state.user_count(),
                        state.node_count()
                    ),
                );
            }
            TuiEvent::SessionsListed { sessions } => {
                self.sessions = sessions.clone();
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayState,
                    &format!("loaded {} sessions", self.sessions.len()),
                );
            }
            TuiEvent::SessionLoaded { session } => {
                self.upsert_session(session.clone());
                self.selected_session_id = Some(session.session_id.clone());
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::GatewayState,
                    &format!("selected session {}", session.session_id),
                );
            }
            TuiEvent::OperationAccepted {
                operation_id,
                context,
            } => {
                self.active_operations
                    .insert(operation_id.clone(), context.clone());
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::Operation,
                    &format!("{context}: accepted ({operation_id})"),
                );
            }
            TuiEvent::OperationCompleted {
                operation_id,
                context,
            } => {
                self.active_operations.remove(operation_id);
                self.push_history(
                    HistorySource::Gateway,
                    HistoryKind::Operation,
                    &format!("{context}: completed ({operation_id})"),
                );
            }
            TuiEvent::OperationFailed {
                operation_id,
                context,
                message,
            } => {
                self.active_operations.remove(operation_id);
                self.push_history(
                    HistorySource::Error,
                    HistoryKind::Error,
                    &format!("{context}: failed ({operation_id}): {message}"),
                );
            }
            TuiEvent::InferenceRunEvent { event } => {
                self.push_inference_event(event);
            }
            TuiEvent::Error { context, message } => {
                self.push_history(
                    HistorySource::Error,
                    HistoryKind::Error,
                    &format!("{context}: {message}"),
                );
            }
        }
    }

    /// Records a user-submitted prompt before it is sent to the engine.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The plain-text prompt submitted by the user.
    pub fn record_user_prompt(&mut self, prompt: &str) {
        self.push_history(HistorySource::User, HistoryKind::UserPrompt, prompt);
    }

    /// Records a user-submitted slash command before it is sent to the engine.
    ///
    /// # Arguments
    ///
    /// * `command` - The slash command text submitted by the user.
    pub fn record_user_command(&mut self, command: &str) {
        self.push_history(HistorySource::User, HistoryKind::UserCommand, command);
    }

    /// Returns the current connection status.
    pub fn connection_status(&self) -> ConnectionStatus {
        self.connection_status
    }

    /// Returns the latest known gateway state snapshot.
    pub fn state(&self) -> Option<&NexoState> {
        self.state.as_ref()
    }

    /// Returns the latest known session list.
    pub fn sessions(&self) -> &[Session] {
        &self.sessions
    }

    /// Returns the currently selected session identifier.
    pub fn selected_session_id(&self) -> Option<&SessionId> {
        self.selected_session_id.as_ref()
    }

    /// Returns the active operations currently tracked by the engine.
    pub fn active_operations(&self) -> &HashMap<OperationId, String> {
        &self.active_operations
    }

    /// Returns the recent event timeline rendered by the UI.
    pub fn timeline(&self) -> &[HistoryEntry] {
        &self.timeline
    }

    /// Pushes a structured history entry onto the timeline.
    ///
    /// # Arguments
    ///
    /// * `source` - The actor responsible for the entry.
    /// * `kind` - The semantic category of the entry.
    /// * `body` - The user-visible body content for the entry.
    fn push_history(&mut self, source: HistorySource, kind: HistoryKind, body: &str) {
        self.timeline.push(HistoryEntry {
            source,
            kind,
            body: body.into(),
        });
    }

    /// Converts an inference event into one or more structured history entries.
    ///
    /// # Arguments
    ///
    /// * `event` - The streaming inference event received from the gateway.
    fn push_inference_event(
        &mut self,
        event: &nexo_ws_schema::NexoEvent<nexo_ws_schema::InferenceRunEvent>,
    ) {
        use nexo_core::InferenceOutputDelta;
        use nexo_ws_schema::InferenceRunEvent;

        match event {
            nexo_ws_schema::NexoEvent::Correlated { event, .. }
            | nexo_ws_schema::NexoEvent::Unsolicited { event } => match event {
                InferenceRunEvent::RunStarted { .. } => {
                    self.push_history(
                        HistorySource::Inference,
                        HistoryKind::Operation,
                        "run started",
                    );
                }
                InferenceRunEvent::RoundCompleted { .. } => {
                    self.push_history(
                        HistorySource::Inference,
                        HistoryKind::Operation,
                        "round completed",
                    );
                }
                InferenceRunEvent::RunCompleted { total_outputs, .. } => {
                    self.push_history(
                        HistorySource::Inference,
                        HistoryKind::Operation,
                        &format!("run completed after {total_outputs:?} output chunks"),
                    );
                }
                InferenceRunEvent::Cancelled { reason, .. } => {
                    self.push_history(
                        HistorySource::Inference,
                        HistoryKind::Operation,
                        reason.as_deref().unwrap_or("run cancelled"),
                    );
                }
                InferenceRunEvent::Failed { error, .. } => {
                    self.push_history(
                        HistorySource::Error,
                        HistoryKind::Error,
                        &format!("inference: {error}"),
                    );
                }
                InferenceRunEvent::Output { output, .. } => match output {
                    InferenceOutputDelta::MultiModal(delta) => {
                        if let Some(reasoning_delta) = &delta.reasoning_delta {
                            self.push_history(
                                HistorySource::Inference,
                                HistoryKind::InferenceThinking,
                                reasoning_delta,
                            );
                        }
                        if let Some(content_delta) = &delta.content_delta {
                            self.push_history(
                                HistorySource::Inference,
                                HistoryKind::InferenceText,
                                content_delta,
                            );
                        }
                        if !delta.tool_call_deltas.is_empty() {
                            self.push_history(
                                HistorySource::Tool,
                                HistoryKind::ToolActivity,
                                &format!("{} tool call updates", delta.tool_call_deltas.len()),
                            );
                        }
                    }
                    other => {
                        self.push_history(
                            HistorySource::Inference,
                            HistoryKind::InferenceText,
                            &format!("{other:?}"),
                        );
                    }
                },
            },
        }
    }

    /// Inserts or replaces a session entry in the cached session list.
    ///
    /// # Arguments
    ///
    /// * `session` - The session entry to insert or replace.
    fn upsert_session(&mut self, session: Session) {
        if let Some(existing_session) = self
            .sessions
            .iter_mut()
            .find(|existing| existing.session_id == session.session_id)
        {
            *existing_session = session;
            return;
        }

        self.sessions.push(session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexo_core::system::node::NodeState;
    use nexo_core::{ClientInfo, DeviceInfo, Node, NodeProperties, User, UserProperties};
    use std::collections::HashSet;

    #[test]
    fn applies_connected_event() {
        let mut state = NexoUserState::new();

        state.apply_event(&TuiEvent::Connected);

        assert_eq!(state.connection_status(), ConnectionStatus::Connected);
    }

    #[test]
    fn tracks_operation_acceptance_and_completion() {
        let mut state = NexoUserState::new();
        let operation_id = OperationId::new();

        state.apply_event(&TuiEvent::OperationAccepted {
            operation_id: operation_id.clone(),
            context: "get_state".into(),
        });
        assert!(state.active_operations().contains_key(&operation_id));

        state.apply_event(&TuiEvent::OperationCompleted {
            operation_id: operation_id.clone(),
            context: "get_state".into(),
        });

        assert!(!state.active_operations().contains_key(&operation_id));
    }

    #[test]
    fn records_user_prompt_with_explicit_source_and_kind() {
        let mut state = NexoUserState::new();

        state.record_user_prompt("hello world");

        let entry = state
            .timeline()
            .last()
            .expect("timeline should contain prompt entry");
        assert_eq!(entry.source, HistorySource::User);
        assert_eq!(entry.kind, HistoryKind::UserPrompt);
        assert_eq!(entry.body, "hello world");
    }

    #[test]
    fn state_updated_history_includes_user_and_node_counts() {
        let mut user_state = NexoUserState::new();
        let mut nexo_state = NexoState::new();

        let user_properties =
            UserProperties::new(ClientInfo::new("user"), DeviceInfo::default(), "token");
        nexo_state
            .add_user(User::from_properties(&user_properties))
            .expect("failed to add user");

        let node_properties =
            NodeProperties::new(ClientInfo::new("node"), DeviceInfo::default(), "token");
        nexo_state
            .add_node(Node::from_properties(
                &node_properties,
                NodeState::Idle,
                HashSet::new(),
            ))
            .expect("failed to add node");

        user_state.apply_event(&TuiEvent::StateUpdated { state: nexo_state });

        let entry = user_state
            .timeline()
            .last()
            .expect("timeline should contain state update");
        assert_eq!(entry.kind, HistoryKind::GatewayState);
        assert_eq!(entry.body, "gateway state updated (users: 1, nodes: 1)");
    }
}
