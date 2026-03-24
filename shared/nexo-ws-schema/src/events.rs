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
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
            status: "streaming".into(),
            content: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("content").is_none());
    }
}
