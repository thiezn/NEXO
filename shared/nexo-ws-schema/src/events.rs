use crate::methods::AgentStatus;
use crate::types::Role;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Server-push event kinds.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    Tick,
    Agent,
    Presence,
    Shutdown,
    Heartbeat,
    Cron,
}

/// Payload for `tick` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TickPayload {
    pub timestamp: String,
    pub seq: u64,
}

/// Payload for `agent` streaming events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentEventPayload {
    pub run_id: String,
    pub session_id: String,
    pub status: AgentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Payload for `presence` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PresencePayload {
    pub client_id: String,
    pub role: Role,
    pub status: String,
}

/// Payload for `shutdown` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ShutdownPayload {
    pub reason: String,
}

/// Payload for `heartbeat` events (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HeartbeatPayload {}

/// Payload for `cron` events.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronPayload {
    pub job_id: String,
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
            serde_json::to_string(&EventKind::Heartbeat).unwrap(),
            "\"heartbeat\""
        );
        assert_eq!(serde_json::to_string(&EventKind::Cron).unwrap(), "\"cron\"");
    }

    #[test]
    fn event_kind_roundtrip() {
        for kind in [
            EventKind::Tick,
            EventKind::Agent,
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
            role: Role::User,
            status: "online".into(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: PresencePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn agent_event_payload_optional_content() {
        let payload = AgentEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: AgentStatus::Streaming,
            content: None,
            tool_name: None,
            tool_call_id: None,
            error: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("content").is_none());
        assert!(json.get("toolName").is_none());
        assert!(json.get("error").is_none());
        assert_eq!(json["status"], "streaming");
        assert_eq!(json["sessionId"], "sess-1");
    }

    #[test]
    fn agent_event_payload_with_tool_fields() {
        let payload = AgentEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: AgentStatus::ToolCall,
            content: None,
            tool_name: Some("echo.run".into()),
            tool_call_id: Some("tc-1".into()),
            error: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["status"], "tool_call");
        assert_eq!(json["toolName"], "echo.run");
        assert_eq!(json["toolCallId"], "tc-1");
    }

    #[test]
    fn agent_event_payload_roundtrip() {
        let payload = AgentEventPayload {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: AgentStatus::Completed,
            content: Some("done".into()),
            tool_name: None,
            tool_call_id: None,
            error: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: AgentEventPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload, decoded);
    }
}
