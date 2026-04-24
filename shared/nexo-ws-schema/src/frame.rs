use crate::error::ErrorPayload;
use crate::events::EventKind;
use crate::methods::Method;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Top-level WebSocket frame envelope.
///
/// Wire format uses `"type"` as the discriminator tag:
/// - `{"type":"request", "id":"...", "method":"...", "params":{...}}`
/// - `{"type":"response", "id":"...", "ok":true, "payload":{...}}`
/// - `{"type":"event", "event":"...", "payload":{...}}`
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Frame {
    Request {
        id: String,
        method: Method,
        params: serde_json::Value,
    },
    Response {
        id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        payload: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorPayload>,
    },
    Event {
        event: EventKind,
        payload: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seq: Option<u64>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "stateVersion"
        )]
        state_version: Option<u64>,
    },
}

impl Frame {
    /// Generate a new time-sortable UUID v7 as a string.
    pub fn new_id() -> String {
        uuid::Uuid::now_v7().to_string()
    }

    /// Build a request frame from a typed method and params.
    pub fn request(method: Method, params: impl Serialize) -> Result<Self, serde_json::Error> {
        Ok(Frame::Request {
            id: Self::new_id(),
            method,
            params: serde_json::to_value(params)?,
        })
    }

    /// Build a successful response frame.
    pub fn ok_response(id: &str, payload: impl Serialize) -> Result<Self, serde_json::Error> {
        Ok(Frame::Response {
            id: id.to_string(),
            ok: true,
            payload: Some(serde_json::to_value(payload)?),
            error: None,
        })
    }

    /// Build an error response frame.
    pub fn error_response(id: &str, error: ErrorPayload) -> Self {
        Frame::Response {
            id: id.to_string(),
            ok: false,
            payload: None,
            error: Some(error),
        }
    }

    /// Build an event frame.
    pub fn event(kind: EventKind, payload: impl Serialize) -> Result<Self, serde_json::Error> {
        Ok(Frame::Event {
            event: kind,
            payload: serde_json::to_value(payload)?,
            seq: None,
            state_version: None,
        })
    }

    /// Build an event frame with a sequence number.
    pub fn event_with_seq(
        kind: EventKind,
        payload: impl Serialize,
        seq: u64,
    ) -> Result<Self, serde_json::Error> {
        Ok(Frame::Event {
            event: kind,
            payload: serde_json::to_value(payload)?,
            seq: Some(seq),
            state_version: None,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn request_serialization() {
        let frame = Frame::Request {
            id: "req-1".into(),
            method: Method::Health,
            params: serde_json::json!({}),
        };
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["type"], "request");
        assert_eq!(json["id"], "req-1");
        assert_eq!(json["method"], "health");
    }

    #[test]
    fn response_ok_serialization() {
        let frame = Frame::ok_response("req-1", serde_json::json!({"status": "ok"})).unwrap();
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["type"], "response");
        assert_eq!(json["ok"], true);
        assert_eq!(json["payload"]["status"], "ok");
        assert!(json.get("error").is_none());
    }

    #[test]
    fn response_error_serialization() {
        let frame = Frame::error_response(
            "req-1",
            ErrorPayload {
                code: "auth_failed".into(),
                message: "bad token".into(),
            },
        );
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["type"], "response");
        assert_eq!(json["ok"], false);
        assert_eq!(json["error"]["code"], "auth_failed");
        assert!(json.get("payload").is_none());
    }

    #[test]
    fn event_serialization() {
        let frame = Frame::event_with_seq(
            EventKind::Tick,
            serde_json::json!({"timestamp": "2026-01-01T00:00:00Z"}),
            1,
        )
        .unwrap();
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["type"], "event");
        assert_eq!(json["event"], "tick");
        assert_eq!(json["seq"], 1);
    }

    #[test]
    fn event_without_seq_omits_field() {
        let frame = Frame::event(EventKind::Heartbeat, serde_json::json!({})).unwrap();
        let json = serde_json::to_value(&frame).unwrap();
        assert!(json.get("seq").is_none());
        assert!(json.get("stateVersion").is_none());
    }

    #[test]
    fn request_roundtrip() {
        let frame = Frame::request(Method::Health, serde_json::json!({})).unwrap();
        let json = serde_json::to_string(&frame).unwrap();
        let decoded: Frame = serde_json::from_str(&json).unwrap();
        // Can't compare directly due to generated ID, but check structure
        if let Frame::Request { method, .. } = decoded {
            assert_eq!(method, Method::Health);
        } else {
            panic!("Expected Request frame");
        }
    }

    #[test]
    fn connect_request_matches_protocol_doc() {
        use crate::connect::ConnectParams;
        use crate::types::*;

        let params = ConnectParams {
            min_protocol: 3,
            max_protocol: 3,
            client: ClientInfo {
                id: "cli".into(),
                version: "1.2.3".into(),
                platform: Platform::Macos,
            },
            role: Role::User,
            scopes: vec![Scope::UserRead, Scope::UserWrite],
            capabilities: vec![],
            commands: vec![],
            models: vec![],
            locale: Some("en-US".into()),
            user_agent: Some("NEXO-cli/1.2.3".into()),
            device: Some(DeviceInfo {
                id: "device_fingerprint".into(),
            }),
        };

        let frame = Frame::Request {
            id: "test-id".into(),
            method: Method::Connect,
            params: serde_json::to_value(&params).unwrap(),
        };

        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(json["type"], "request");
        assert_eq!(json["method"], "connect");
        assert_eq!(json["params"]["minProtocol"], 3);
        assert_eq!(json["params"]["client"]["id"], "cli");
        assert_eq!(json["params"]["role"], "user");
        assert_eq!(json["params"]["scopes"][0], "user.read");
    }

    #[test]
    fn hello_ok_response_matches_protocol_doc() {
        use crate::connect::HelloOk;

        let hello = HelloOk::default();
        let frame = Frame::ok_response("test-id", &hello).unwrap();
        let json = serde_json::to_value(&frame).unwrap();

        assert_eq!(json["type"], "response");
        assert_eq!(json["ok"], true);
        assert_eq!(json["payload"]["type"], "hello-ok");
        assert_eq!(json["payload"]["protocol"], 3);
        assert_eq!(json["payload"]["policy"]["tickIntervalMs"], 15000);
    }

    #[test]
    fn new_id_is_valid_uuid() {
        let id = Frame::new_id();
        assert_eq!(id.len(), 36);
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }

    #[test]
    fn deserialization_from_raw_json() {
        let raw = r#"{"type":"request","id":"abc","method":"health","params":{}}"#;
        let frame: Frame = serde_json::from_str(raw).unwrap();
        if let Frame::Request { id, method, .. } = frame {
            assert_eq!(id, "abc");
            assert_eq!(method, Method::Health);
        } else {
            panic!("Expected Request");
        }
    }
}
