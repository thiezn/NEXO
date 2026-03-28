use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Available request methods in the gateway protocol.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    Connect,
    Health,
    Status,
    Send,
    Agent,
    SystemPresence,
    #[serde(rename = "tools.catalog")]
    ToolsCatalog,
    #[serde(rename = "tools.register")]
    ToolsRegister,
    #[serde(rename = "tools.execute")]
    ToolsExecute,
    #[serde(rename = "session.create")]
    SessionCreate,
    #[serde(rename = "session.list")]
    SessionList,
    #[serde(rename = "session.get")]
    SessionGet,
    #[serde(rename = "session.clear")]
    SessionClear,
    #[serde(rename = "cron.create")]
    CronCreate,
    #[serde(rename = "cron.list")]
    CronList,
    #[serde(rename = "cron.delete")]
    CronDelete,
}

/// Agent run status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Accepted,
    Thinking,
    ToolCall,
    Streaming,
    Completed,
    Failed,
    Cancelled,
}

// -- Request param types --

/// Parameters for the `health` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HealthParams {}

/// Parameters for the `status` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct StatusParams {}

/// Parameters for the `send` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SendParams {
    pub target: String,
    pub payload: serde_json::Value,
    pub idempotency_key: String,
}

/// Parameters for the `agent` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentParams {
    pub prompt: String,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
}

/// Parameters for the `system-presence` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SystemPresenceParams {
    pub status: String,
}

/// Parameters for the `tools.catalog` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsCatalogParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
}

// -- Response payload types --

/// Response payload for `health`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
}

/// Response payload for `status`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StatusResponse {
    pub connected_users: u32,
    pub connected_nodes: u32,
    pub capabilities: Vec<String>,
}

/// Response payload for `send`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SendResponse {
    pub delivered: bool,
}

/// Response payload for `agent` (initial ack and final result).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentResponse {
    pub run_id: String,
    pub session_id: String,
    pub status: AgentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// A single tool entry in the tools catalog response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolEntry {
    pub name: String,
    pub description: String,
    pub source: String,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Response payload for `tools.catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsCatalogResponse {
    pub tools: Vec<ToolEntry>,
}

// -- tools.register --

/// A tool specification entry for registration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolSpecEntry {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Parameters for the `tools.register` method (sent by nodes).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsRegisterParams {
    pub tools: Vec<ToolSpecEntry>,
}

/// Response payload for `tools.register`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsRegisterResponse {
    pub registered: u32,
}

// -- tools.execute --

/// Parameters for the `tools.execute` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolsExecuteParams {
    pub tool: String,
    pub args: serde_json::Value,
    pub idempotency_key: String,
}

/// Response payload for `tools.execute`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ToolsExecuteResponse {
    pub success: bool,
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// -- session.create --

/// Parameters for the `session.create` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionCreateParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Response payload for `session.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateResponse {
    pub session_id: String,
}

// -- session.list --

/// Parameters for the `session.list` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionListParams {}

/// A single session entry in a session list response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntry {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub created_at: String,
    pub last_active_at: String,
    pub message_count: u32,
}

/// Response payload for `session.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionListResponse {
    pub sessions: Vec<SessionEntry>,
}

// -- session.get --

/// Parameters for the `session.get` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionGetParams {
    pub session_id: String,
}

/// A single conversation message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Response payload for `session.get`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionGetResponse {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub created_at: String,
}

// -- session.clear --

/// Parameters for the `session.clear` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionClearParams {
    pub session_id: String,
}

/// Response payload for `session.clear`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionClearResponse {
    pub cleared: bool,
}

// -- cron.create --

/// Parameters for the `cron.create` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronCreateParams {
    pub name: String,
    pub schedule: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Response payload for `cron.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronCreateResponse {
    pub job_id: String,
}

// -- cron.list --

/// Parameters for the `cron.list` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronListParams {}

/// A single cron job entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronEntry {
    pub job_id: String,
    pub name: String,
    pub schedule: String,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_run_at: Option<String>,
}

/// Response payload for `cron.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronListResponse {
    pub jobs: Vec<CronEntry>,
}

// -- cron.delete --

/// Parameters for the `cron.delete` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronDeleteParams {
    pub job_id: String,
}

/// Response payload for `cron.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronDeleteResponse {
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn method_serialization() {
        assert_eq!(serde_json::to_string(&Method::Connect).unwrap(), "\"connect\"");
        assert_eq!(serde_json::to_string(&Method::Health).unwrap(), "\"health\"");
        assert_eq!(
            serde_json::to_string(&Method::SystemPresence).unwrap(),
            "\"system-presence\""
        );
        assert_eq!(
            serde_json::to_string(&Method::ToolsCatalog).unwrap(),
            "\"tools.catalog\""
        );
    }

    #[test]
    fn method_deserialization() {
        let m: Method = serde_json::from_str("\"tools.catalog\"").unwrap();
        assert_eq!(m, Method::ToolsCatalog);

        let m: Method = serde_json::from_str("\"system-presence\"").unwrap();
        assert_eq!(m, Method::SystemPresence);
    }

    #[test]
    fn send_params_camel_case() {
        let params = SendParams {
            target: "node-1".into(),
            payload: serde_json::json!({"data": "hello"}),
            idempotency_key: "key-123".into(),
        };
        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["idempotencyKey"], "key-123");
    }

    #[test]
    fn agent_params_roundtrip() {
        let params = AgentParams {
            prompt: "summarize this".into(),
            idempotency_key: "idem-1".into(),
            session_id: Some("sess-1".into()),
            context: Some(serde_json::json!({"files": ["a.rs"]})),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: AgentParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn agent_params_without_session_omits_field() {
        let params = AgentParams {
            prompt: "hello".into(),
            idempotency_key: "k1".into(),
            session_id: None,
            context: None,
        };
        let json = serde_json::to_value(&params).unwrap();
        assert!(!json.as_object().unwrap().contains_key("sessionId"));
    }

    #[test]
    fn health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".into(),
            uptime_secs: 3600,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["uptimeSecs"], 3600);
    }

    #[test]
    fn tools_catalog_response() {
        let resp = ToolsCatalogResponse {
            tools: vec![ToolEntry {
                name: "extractor".into(),
                description: "Extract data".into(),
                source: "core".into(),
                available: true,
                parameters: None,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: ToolsCatalogResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.tools.len(), decoded.tools.len());
        assert_eq!(resp.tools[0].name, decoded.tools[0].name);
    }

    #[test]
    fn tool_entry_with_parameters() {
        let entry = ToolEntry {
            name: "echo".into(),
            description: "Echo input".into(),
            source: "node".into(),
            available: true,
            parameters: Some(serde_json::json!({"type": "object"})),
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["parameters"]["type"], "object");
    }

    #[test]
    fn tool_entry_without_parameters_omits_field() {
        let entry = ToolEntry {
            name: "echo".into(),
            description: "Echo input".into(),
            source: "node".into(),
            available: true,
            parameters: None,
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert!(!json.as_object().unwrap().contains_key("parameters"));
    }

    #[test]
    fn tools_register_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::ToolsRegister).unwrap(),
            "\"tools.register\""
        );
        assert_eq!(
            serde_json::to_string(&Method::ToolsExecute).unwrap(),
            "\"tools.execute\""
        );
    }

    #[test]
    fn tools_register_params_roundtrip() {
        let params = ToolsRegisterParams {
            tools: vec![ToolSpecEntry {
                name: "echo".into(),
                description: "Echo tool".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"input": {"type": "string"}}}),
            }],
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: ToolsRegisterParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn tools_execute_params_camel_case() {
        let params = ToolsExecuteParams {
            tool: "echo".into(),
            args: serde_json::json!({"input": "hello"}),
            idempotency_key: "key-1".into(),
        };
        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["idempotencyKey"], "key-1");
    }

    #[test]
    fn tools_execute_response_roundtrip() {
        let resp = ToolsExecuteResponse {
            success: true,
            output: "hello".into(),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: ToolsExecuteResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn agent_status_serialization() {
        for (status, expected) in [
            (AgentStatus::Accepted, "\"accepted\""),
            (AgentStatus::Thinking, "\"thinking\""),
            (AgentStatus::ToolCall, "\"tool_call\""),
            (AgentStatus::Streaming, "\"streaming\""),
            (AgentStatus::Completed, "\"completed\""),
            (AgentStatus::Failed, "\"failed\""),
            (AgentStatus::Cancelled, "\"cancelled\""),
        ] {
            let json = serde_json::to_string(&status).unwrap();
            assert_eq!(json, expected);
            let decoded: AgentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, decoded);
        }
    }

    #[test]
    fn agent_response_with_typed_status() {
        let resp = AgentResponse {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: AgentStatus::Accepted,
            summary: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["runId"], "run-1");
        assert_eq!(json["sessionId"], "sess-1");
        assert_eq!(json["status"], "accepted");
        assert!(json.get("summary").is_none());
    }

    #[test]
    fn session_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::SessionCreate).unwrap(),
            "\"session.create\""
        );
        assert_eq!(
            serde_json::to_string(&Method::SessionList).unwrap(),
            "\"session.list\""
        );
        assert_eq!(
            serde_json::to_string(&Method::SessionGet).unwrap(),
            "\"session.get\""
        );
        assert_eq!(
            serde_json::to_string(&Method::SessionClear).unwrap(),
            "\"session.clear\""
        );
    }

    #[test]
    fn cron_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::CronCreate).unwrap(),
            "\"cron.create\""
        );
        assert_eq!(
            serde_json::to_string(&Method::CronList).unwrap(),
            "\"cron.list\""
        );
        assert_eq!(
            serde_json::to_string(&Method::CronDelete).unwrap(),
            "\"cron.delete\""
        );
    }

    #[test]
    fn session_create_params_roundtrip() {
        let params = SessionCreateParams {
            name: Some("my session".into()),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: SessionCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn session_entry_camel_case() {
        let entry = SessionEntry {
            session_id: "s1".into(),
            name: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            last_active_at: "2026-01-01T01:00:00Z".into(),
            message_count: 5,
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["lastActiveAt"], "2026-01-01T01:00:00Z");
        assert_eq!(json["messageCount"], 5);
    }

    #[test]
    fn cron_create_params_roundtrip() {
        let params = CronCreateParams {
            name: "daily summary".into(),
            schedule: "0 9 * * *".into(),
            prompt: "summarize yesterday".into(),
            session_id: Some("sess-1".into()),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: CronCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn cron_entry_optional_fields() {
        let entry = CronEntry {
            job_id: "j1".into(),
            name: "test".into(),
            schedule: "* * * * *".into(),
            enabled: true,
            last_run_at: None,
            next_run_at: Some("2026-01-01T00:00:00Z".into()),
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert!(json.get("lastRunAt").is_none());
        assert_eq!(json["nextRunAt"], "2026-01-01T00:00:00Z");
    }

    #[test]
    fn conversation_message_roundtrip() {
        let msg = ConversationMessage {
            id: "m1".into(),
            role: "assistant".into(),
            content: "hello".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            tool_call_id: None,
            tool_name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: ConversationMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, decoded);
    }
}
