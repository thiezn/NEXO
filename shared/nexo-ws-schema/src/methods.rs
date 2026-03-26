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
    pub status: String,
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
            context: Some(serde_json::json!({"files": ["a.rs"]})),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: AgentParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
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
}
