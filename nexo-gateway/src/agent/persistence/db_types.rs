use nexo_core::MessageRole;
use nexo_ws_schema::RunStatus;

/// Persisted classification for conversation entry rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationEntryKind {
    /// End-user authored text input.
    UserInput,
    /// System or developer instructions.
    Instruction,
    /// Assistant-authored visible response text.
    AssistantResponse,
    /// Assistant-authored tool call plan.
    ToolCallIntent,
    /// Tool execution output injected into the transcript.
    ToolResult,
}

impl ConversationEntryKind {
    /// Return the SQLite string representation for the entry kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserInput => "user_input",
            Self::Instruction => "instruction",
            Self::AssistantResponse => "assistant_response",
            Self::ToolCallIntent => "tool_call_intent",
            Self::ToolResult => "tool_result",
        }
    }

    /// Infer the persisted entry kind from the semantic message role.
    pub const fn from_role(role: MessageRole) -> Self {
        match role {
            MessageRole::System | MessageRole::Developer => Self::Instruction,
            MessageRole::User => Self::UserInput,
            MessageRole::Assistant => Self::AssistantResponse,
            MessageRole::Tool => Self::ToolResult,
        }
    }
}

impl TryFrom<&str> for ConversationEntryKind {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "user_input" => Ok(Self::UserInput),
            "instruction" => Ok(Self::Instruction),
            "assistant_response" => Ok(Self::AssistantResponse),
            "tool_call_intent" => Ok(Self::ToolCallIntent),
            "tool_result" => Ok(Self::ToolResult),
            _ => Err(format!("invalid conversation entry kind '{value}'")),
        }
    }
}

/// Lifecycle state persisted for a run round row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundStatus {
    /// The round has been created but not yet finished.
    Started,
    /// The round completed normally.
    Completed,
    /// The round failed.
    Failed,
    /// The round was queued waiting for capacity.
    Queued,
    /// The round was cancelled.
    Cancelled,
}

impl RoundStatus {
    /// Return the SQLite string representation for the round status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Queued => "queued",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Lifecycle state persisted for a tool trace row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTraceStatus {
    /// The trace has been created but not yet finished.
    Started,
    /// The tool finished successfully.
    Completed,
    /// The tool finished with an error.
    Failed,
}

impl ToolTraceStatus {
    /// Return the SQLite string representation for the tool trace status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    /// Map a boolean success flag to the corresponding terminal trace state.
    pub const fn from_success(success: bool) -> Self {
        if success {
            Self::Completed
        } else {
            Self::Failed
        }
    }
}

/// Persisted classification for run summary rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunSummaryKind {
    /// Final assistant response content.
    FinalResponse,
    /// Failure summary content.
    Failure,
    /// Cancellation summary content.
    Cancelled,
    /// Catch-all terminal summary classification.
    TerminalState,
}

impl RunSummaryKind {
    /// Return the SQLite string representation for the summary kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FinalResponse => "final_response",
            Self::Failure => "failure",
            Self::Cancelled => "cancelled",
            Self::TerminalState => "terminal_state",
        }
    }

    /// Derive the persisted summary kind for a terminal run status.
    pub const fn from_terminal_run_status(status: RunStatus) -> Self {
        match status {
            RunStatus::Completed => Self::FinalResponse,
            RunStatus::Failed => Self::Failure,
            RunStatus::Cancelled => Self::Cancelled,
            _ => Self::TerminalState,
        }
    }
}

pub(crate) fn parse_message_role(value: &str) -> Result<MessageRole, sqlx::Error> {
    match value {
        "system" => Ok(MessageRole::System),
        "developer" => Ok(MessageRole::Developer),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        _ => Err(decode_enum_error("message role", value)),
    }
}

pub(crate) fn parse_entry_kind(value: &str) -> Result<ConversationEntryKind, sqlx::Error> {
    ConversationEntryKind::try_from(value).map_err(|_| decode_enum_error("entry kind", value))
}

pub(crate) const fn message_role_to_db(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::Developer => "developer",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Return the SQLite string representation for the wire-level run status.
pub const fn run_status_to_db(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Accepted => "accepted",
        RunStatus::Queued => "queued",
        RunStatus::Thinking => "thinking",
        RunStatus::ToolCall => "tool_call",
        RunStatus::Streaming => "streaming",
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

fn decode_enum_error(field: &'static str, value: &str) -> sqlx::Error {
    sqlx::Error::Decode(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid {field}: {value}"),
    )))
}
