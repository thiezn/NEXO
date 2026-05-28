use crate::methods::RunStatus;
use crate::types::ConnectionRole;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Server-push event kinds.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    /// Periodic heartbeat tick event.
    Tick,
    /// Run lifecycle update event.
    Run,
    /// Message delivery event.
    Message,
    /// Presence state update event.
    Presence,
    /// Server shutdown notification event.
    Shutdown,
    /// Keepalive heartbeat event.
    Heartbeat,
    /// Cron job execution event.
    Cron,
}

/// Payload for `tick` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TickPayload {
    /// Timestamp at which the tick was emitted.
    pub timestamp: String,
    /// Monotonic sequence number for the tick stream.
    pub seq: u64,
}

/// Payload for `run` streaming events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunEventPayload {
    /// Run identifier.
    pub run_id: String,
    /// Session identifier associated with the run.
    pub session_id: String,
    /// Current run status.
    pub status: RunStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional textual output content.
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional tool name when status reflects a tool call.
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional tool call identifier when status reflects a tool call.
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional error message when the run fails.
    pub error: Option<String>,
    /// Ephemeral thinking/reasoning content (not persisted in conversation history).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_content: Option<String>,
}

/// Payload for `presence` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PresencePayload {
    /// Client identifier whose status changed.
    pub client_id: String,
    /// Role of the client.
    pub role: ConnectionRole,
    /// Presence status value.
    pub status: String,
}

/// Payload for `message` events delivered by the gateway.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MessagePayload {
    /// Message identifier.
    pub message_id: String,
    /// Message sender identifier.
    pub from: String,
    /// Message target identifier.
    pub target: String,
    /// Arbitrary message payload data.
    pub payload: serde_json::Value,
}

/// Payload for `shutdown` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ShutdownPayload {
    /// Human-readable shutdown reason.
    pub reason: String,
}

/// Payload for `heartbeat` events (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HeartbeatPayload {}

/// Payload for `cron` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronPayload {
    /// Cron job identifier.
    pub job_id: String,
    /// Cron job display name.
    pub name: String,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn event_kind_serialization() {
        assert_eq!(serde_json::to_string(&EventKind::Tick).unwrap(), "\"tick\"");
        assert_eq!(
            serde_json::to_string(&EventKind::Message).unwrap(),
            "\"message\""
        );
        assert_eq!(
            serde_json::to_string(&EventKind::Heartbeat).unwrap(),
            "\"heartbeat\""
        );
        assert_eq!(serde_json::to_string(&EventKind::Cron).unwrap(), "\"cron\"");
    }

    #[test]
    fn event_kind_roundtrip() {
        for kind in [
            EventKind::Tick,
            EventKind::Run,
            EventKind::Message,
            EventKind::Presence,
            EventKind::Shutdown,
            EventKind::Heartbeat,
            EventKind::Cron,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let decoded: EventKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, decoded);
        }
    }

    #[test]
    fn tick_payload_serialization() {
        let payload = TickPayload {
            timestamp: "2026-03-23T12:00:00Z".into(),
            seq: 42,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["seq"], 42);
    }

    #[test]
    fn presence_payload_roundtrip() {
        let payload = PresencePayload {
            client_id: "cli-1".into(),
            role: ConnectionRole::User,
            status: "online".into(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: PresencePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn message_payload_roundtrip() {
        let payload = MessagePayload {
            message_id: "msg-1".into(),
            from: "user-a".into(),
            target: "user-b".into(),
            payload: serde_json::json!({"text": "hello"}),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: MessagePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn run_event_payload_optional_content() {
        let payload = RunEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: RunStatus::Streaming,
            content: None,
            tool_name: None,
            tool_call_id: None,
            error: None,
            thinking_content: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("content").is_none());
        assert!(json.get("toolName").is_none());
        assert!(json.get("error").is_none());
        assert_eq!(json["status"], "streaming");
        assert_eq!(json["sessionId"], "sess-1");
    }

    #[test]
    fn run_event_payload_with_tool_fields() {
        let payload = RunEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: RunStatus::ToolCall,
            content: None,
            tool_name: Some("echo.run".into()),
            tool_call_id: Some("tc-1".into()),
            error: None,
            thinking_content: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["status"], "tool_call");
        assert_eq!(json["toolName"], "echo.run");
        assert_eq!(json["toolCallId"], "tc-1");
    }

    #[test]
    fn run_event_payload_roundtrip() {
        let payload = RunEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: RunStatus::Completed,
            content: Some("done".into()),
            tool_name: None,
            tool_call_id: None,
            error: None,
            thinking_content: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: RunEventPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }
}
