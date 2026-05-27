use serde::{Deserialize, Serialize};

use crate::message::TranscriptMessage;

/// Semantic kind assigned to a persisted transcript entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TranscriptEntryKind {
    UserInput,
    Instruction,
    AssistantResponse,
    ToolCallIntent,
    ToolResult,
}

impl TranscriptEntryKind {
    /// Return the canonical persisted snake_case representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserInput => "user_input",
            Self::Instruction => "instruction",
            Self::AssistantResponse => "assistant_response",
            Self::ToolCallIntent => "tool_call_intent",
            Self::ToolResult => "tool_result",
        }
    }
}

/// A persisted transcript entry with server-assigned metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct TranscriptEntry {
    pub id: String,
    #[serde(flatten)]
    pub message: TranscriptMessage,
    pub kind: TranscriptEntryKind,
    pub created_at: String,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::message::MessageRole;

    #[test]
    fn transcript_entry_kind_serde_roundtrip() {
        for kind in [
            TranscriptEntryKind::UserInput,
            TranscriptEntryKind::Instruction,
            TranscriptEntryKind::AssistantResponse,
            TranscriptEntryKind::ToolCallIntent,
            TranscriptEntryKind::ToolResult,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let parsed: TranscriptEntryKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, parsed);
        }
    }

    #[test]
    fn transcript_entry_flattens_message_fields() {
        let entry = TranscriptEntry {
            id: "msg-1".into(),
            message: TranscriptMessage::with_tool_metadata(
                MessageRole::Tool,
                "stdout: hello",
                Some("call-1".into()),
                Some("io.bash".into()),
            ),
            kind: TranscriptEntryKind::ToolResult,
            created_at: "2026-05-23T16:00:00Z".into(),
        };

        let json = serde_json::to_value(&entry).unwrap();

        assert_eq!(json["id"], "msg-1");
        assert_eq!(json["role"], "tool");
        assert_eq!(json["toolCallId"], "call-1");
        assert_eq!(json["toolName"], "io.bash");
        assert_eq!(json["kind"], "tool_result");
        assert_eq!(json["createdAt"], "2026-05-23T16:00:00Z");
    }
}
