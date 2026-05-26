use nexo_spec::message::{MessageRole, TranscriptMessage};
#[cfg(test)]
use nexo_spec::tool::ToolExecutionConstraints;
#[cfg(test)]
use nexo_spec::transcript::TranscriptEntryKind;
use nexo_spec::{
    model::LoadedModelInfo, prompt::PromptCollection, tool::ToolSpec, transcript::TranscriptEntry,
};
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
    #[serde(rename = "run.start")]
    RunStart,
    #[serde(rename = "run.stop")]
    RunStop,
    #[serde(rename = "run.instructions.append")]
    RunInstructionsAppend,
    #[serde(rename = "run.round")]
    RunRound,
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
    /// Gateway → node: load a model into VRAM.
    #[serde(rename = "model.load")]
    ModelLoad,
    /// Gateway → node: unload a model from VRAM.
    #[serde(rename = "model.unload")]
    ModelUnload,
    /// Node → gateway: report current loaded model and available models.
    #[serde(rename = "model.status")]
    ModelStatus,
    /// Client → gateway: create a prompt document.
    #[serde(rename = "prompt.document.create")]
    PromptDocumentCreate,
    /// Client → gateway: list prompt documents.
    #[serde(rename = "prompt.document.list")]
    PromptDocumentList,
    /// Client → gateway: delete a prompt document.
    #[serde(rename = "prompt.document.delete")]
    PromptDocumentDelete,
    /// Client → gateway: create a prompt collection.
    #[serde(rename = "prompt.collection.create")]
    PromptCollectionCreate,
    /// Client → gateway: list prompt collections.
    #[serde(rename = "prompt.collection.list")]
    PromptCollectionList,
    /// Client → gateway: delete a prompt collection.
    #[serde(rename = "prompt.collection.delete")]
    PromptCollectionDelete,
    /// Client → gateway → node: analyze an image using the model's vision capabilities.
    #[serde(rename = "image.analyze")]
    ImageAnalyze,
}

/// Run lifecycle status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Accepted,
    /// Run is queued, waiting for an LLM node to become available.
    Queued,
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
#[serde(rename_all = "lowercase")]
pub struct SendParams {
    pub target: String,
    pub payload: serde_json::Value,
    pub idempotency_key: String,
}

/// Parameters for the `run.start` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunStartParams {
    pub input: String,
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<serde_json::Value>,
    /// The model ID to use for inference. If omitted, any available LLM node is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Enable thinking mode (Gemma 4). When true the model emits reasoning
    /// tokens that are returned in the event but not persisted in history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
}

/// Parameters for the `run.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunStopParams {
    /// The active run to stop.
    pub run_id: String,
}

/// Response payload for the `run.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunStopResponse {
    /// Whether the run was still active and has now been stopped.
    pub stopped: bool,
}

/// Parameters for the `run.instructions.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunInstructionsAppendParams {
    /// The active run that should observe the new context on its next round.
    pub run_id: String,
    /// Arbitrary structured instructions to append to the transcript.
    pub instructions: serde_json::Value,
}

/// Response payload for the `run.instructions.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunInstructionsAppendResponse {
    /// Whether the instructions were accepted for the active run.
    pub queued: bool,
    /// The persisted message identifier, when context was queued successfully.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
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
#[serde(rename_all = "lowercase")]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
}

/// Response payload for `status`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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

/// Response payload for `run.start` (initial ack and final result).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunStartResponse {
    pub run_id: String,
    pub session_id: String,
    pub status: RunStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Gateway-to-node payload for a single run round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunRoundRequest {
    pub run_id: String,
    pub round_id: String,
    pub session_id: String,
    pub messages: Vec<TranscriptMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpecEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

/// A single tool call returned from a node for a round.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunRoundToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Node-to-gateway response payload for a single run round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct RunRoundResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<RunRoundToolCall>,
}

/// A single tool entry in the tools catalog response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ToolEntry {
    #[serde(flatten)]
    pub spec: ToolSpec,
    pub source: String,
    pub available: bool,
}

impl ToolEntry {
    /// Create a tool catalog entry from a shared tool spec.
    pub fn new(spec: ToolSpec, source: impl Into<String>, available: bool) -> Self {
        Self {
            spec,
            source: source.into(),
            available,
        }
    }
}

/// Response payload for `tools.catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsCatalogResponse {
    pub tools: Vec<ToolEntry>,
}

// -- tools.register --

/// A tool specification entry for registration.
pub type ToolSpecEntry = ToolSpec;

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
#[serde(rename_all = "lowercase")]
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
#[serde(rename_all = "lowercase")]
pub struct SessionCreateParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// ID of a prompt collection to associate with this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_collection_id: Option<String>,
}

/// Response payload for `session.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct SessionCreateResponse {
    pub session_id: String,
    /// The prompt collection ID associated with this session, if one was provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_collection_id: Option<String>,
}

// -- session.list --

/// Parameters for the `session.list` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionListParams {}

/// A single session entry in a session list response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct SessionEntry {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_collection_id: Option<String>,
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
#[serde(rename_all = "lowercase")]
pub struct SessionGetParams {
    pub session_id: String,
}

/// Response payload for `session.get`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct SessionGetResponse {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_collection_id: Option<String>,
    pub messages: Vec<TranscriptEntry>,
    pub created_at: String,
}

// -- session.clear --

/// Parameters for the `session.clear` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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
#[serde(rename_all = "lowercase")]
pub struct CronCreateParams {
    pub name: String,
    pub schedule: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Response payload for `cron.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct CronCreateResponse {
    pub job_id: String,
}

// -- cron.list --

/// Parameters for the `cron.list` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronListParams {}

/// A single cron job entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
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
#[serde(rename_all = "lowercase")]
pub struct CronDeleteParams {
    pub job_id: String,
}

/// Response payload for `cron.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronDeleteResponse {
    pub deleted: bool,
}

// -- model.load --

/// Parameters for `model.load` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ModelLoadParams {
    pub model_id: String,
}

/// Response payload for `model.load`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ModelLoadResponse {
    pub model_id: String,
    pub loaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// -- model.unload --

/// Parameters for `model.unload` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ModelUnloadParams {
    pub model_id: String,
}

/// Response payload for `model.unload`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ModelUnloadResponse {
    pub unloaded: bool,
}

// -- model.status --

/// Sent by a node to report its model state.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ModelStatusParams {
    /// Models currently loaded with their categories.
    #[serde(default)]
    pub loaded_models: Vec<LoadedModelInfo>,
    /// All model IDs available on disk on this node.
    #[serde(default)]
    pub available_models: Vec<String>,
}

// -- prompt.document.create --

/// Parameters for `prompt.document.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentCreateParams {
    /// Stable ID for the prompt document (for example `identity.md`).
    pub id: String,
    pub content: String,
}

/// Response payload for `prompt.document.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentCreateResponse {
    pub id: String,
}

// -- prompt.document.list --

/// Parameters for `prompt.document.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentListParams {}

/// A single prompt document entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentEntry {
    pub id: String,
}

/// Response payload for `prompt.document.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentListResponse {
    pub documents: Vec<PromptDocumentEntry>,
}

// -- prompt.document.delete --

/// Parameters for `prompt.document.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentDeleteParams {
    pub id: String,
}

/// Response payload for `prompt.document.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentDeleteResponse {
    pub deleted: bool,
}

// -- prompt.collection.create --

/// Parameters for `prompt.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct PromptCollectionCreateParams {
    /// Unique ID for the collection.
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered list of prompt document IDs that form this collection.
    pub documents: Vec<String>,
}

/// Response payload for `prompt.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionCreateResponse {
    pub id: String,
}

// -- prompt.collection.list --

/// Parameters for `prompt.collection.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionListParams {}

/// Response payload for `prompt.collection.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionListResponse {
    pub collections: Vec<PromptCollection>,
}

// -- prompt.collection.delete --

/// Parameters for `prompt.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionDeleteParams {
    pub id: String,
}

/// Response payload for `prompt.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionDeleteResponse {
    pub deleted: bool,
}

// -- image.analyze --

/// Parameters for `image.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
pub struct ImageAnalyzeParams {
    /// Base64-encoded image data.
    pub image_data: String,
    /// The prompt/question about the image.
    pub prompt: String,
    #[serde(default = "default_image_analyze_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_image_analyze_temperature")]
    pub temperature: f64,
    /// Visual token budget for variable resolution. Common values: 70, 140, 280, 560, 1120.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual_token_budget: Option<u32>,
    pub idempotency_key: String,
}

fn default_image_analyze_max_tokens() -> usize {
    4096
}

fn default_image_analyze_temperature() -> f64 {
    1.0
}

/// Response payload for `image.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct ImageAnalyzeResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::Connect).unwrap(),
            "\"connect\""
        );
        assert_eq!(
            serde_json::to_string(&Method::Health).unwrap(),
            "\"health\""
        );
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
    fn run_start_params_roundtrip() {
        let params = RunStartParams {
            input: "summarize this".into(),
            idempotency_key: "idem-1".into(),
            session_id: Some("sess-1".into()),
            instructions: Some(serde_json::json!({"files": ["a.rs"]})),
            model_id: None,
            thinking: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: RunStartParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn run_start_params_without_session_omits_field() {
        let params = RunStartParams {
            input: "hello".into(),
            idempotency_key: "k1".into(),
            session_id: None,
            instructions: None,
            model_id: None,
            thinking: None,
        };
        let json = serde_json::to_value(&params).unwrap();
        assert!(!json.as_object().unwrap().contains_key("sessionId"));
    }

    #[test]
    fn run_control_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::RunStop).unwrap(),
            "\"run.stop\""
        );
        assert_eq!(
            serde_json::to_string(&Method::RunInstructionsAppend).unwrap(),
            "\"run.instructions.append\""
        );
    }

    #[test]
    fn run_stop_params_roundtrip() {
        let params = RunStopParams {
            run_id: "run-1".into(),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: RunStopParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn run_instructions_append_params_roundtrip() {
        let params = RunInstructionsAppendParams {
            run_id: "run-1".into(),
            instructions: serde_json::json!({"files": ["notes.md"]}),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: RunInstructionsAppendParams = serde_json::from_str(&json).unwrap();
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
            tools: vec![ToolEntry::new(
                ToolSpec {
                    name: "extractor".into(),
                    description: "Extract data".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: None,
                    execution: ToolExecutionConstraints::default(),
                },
                "core",
                true,
            )],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: ToolsCatalogResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.tools.len(), decoded.tools.len());
        assert_eq!(resp.tools[0].spec.name, decoded.tools[0].spec.name);
    }

    #[test]
    fn tool_entry_with_parameters() {
        let entry = ToolEntry::new(
            ToolSpec {
                name: "echo".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: Some("2026-05-22".into()),
                execution: ToolExecutionConstraints {
                    side_effect_level: nexo_spec::tool::ToolSideEffectLevel::ReadOnly,
                    parallel_safe: true,
                },
            },
            "node",
            true,
        );
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["parameters"]["type"], "object");
        assert_eq!(json["contractVersion"], "2026-05-22");
        assert_eq!(json["execution"]["sideEffectLevel"], "read_only");
        assert_eq!(json["execution"]["parallelSafe"], true);
    }

    #[test]
    fn tool_entry_omits_default_spec_metadata() {
        let entry = ToolEntry::new(
            ToolSpec {
                name: "echo".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: None,
                execution: ToolExecutionConstraints::default(),
            },
            "node",
            true,
        );
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["parameters"]["type"], "object");
        assert!(!json.as_object().unwrap().contains_key("contractVersion"));
        assert!(!json.as_object().unwrap().contains_key("execution"));
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
                contract_version: Some("2026-05-22".into()),
                execution: ToolExecutionConstraints::default(),
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
    fn run_status_serialization() {
        for (status, expected) in [
            (RunStatus::Accepted, "\"accepted\""),
            (RunStatus::Queued, "\"queued\""),
            (RunStatus::Thinking, "\"thinking\""),
            (RunStatus::ToolCall, "\"tool_call\""),
            (RunStatus::Streaming, "\"streaming\""),
            (RunStatus::Completed, "\"completed\""),
            (RunStatus::Failed, "\"failed\""),
            (RunStatus::Cancelled, "\"cancelled\""),
        ] {
            let json = serde_json::to_string(&status).unwrap();
            assert_eq!(json, expected);
            let decoded: RunStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, decoded);
        }
    }

    #[test]
    fn run_start_response_with_typed_status() {
        let resp = RunStartResponse {
            run_id: "run-1".into(),
            session_id: "sess-1".into(),
            status: RunStatus::Accepted,
            summary: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["runId"], "run-1");
        assert_eq!(json["sessionId"], "sess-1");
        assert_eq!(json["status"], "accepted");
        assert!(json.get("summary").is_none());
    }

    #[test]
    fn run_round_request_roundtrip() {
        let request = RunRoundRequest {
            run_id: "run-1".into(),
            round_id: "round-1".into(),
            session_id: "sess-1".into(),
            messages: vec![TranscriptMessage::new(
                MessageRole::System,
                "You are helpful",
            )],
            tools: vec![ToolSpecEntry {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: None,
                execution: ToolExecutionConstraints::default(),
            }],
            model_id: Some("gemma-4-e4b-it".into()),
        };

        let json = serde_json::to_string(&request).unwrap();
        let decoded: RunRoundRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request, decoded);
    }

    #[test]
    fn run_round_response_roundtrip() {
        let response = RunRoundResponse {
            content: Some("Final answer".into()),
            rationale: Some("Reasoning summary".into()),
            tool_calls: vec![RunRoundToolCall {
                id: "call-1".into(),
                name: "echo.run".into(),
                arguments: serde_json::json!({"input": "hello"}),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        let decoded: RunRoundResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response, decoded);
    }

    #[test]
    fn run_instructions_append_response_omits_missing_message_id() {
        let response = RunInstructionsAppendResponse {
            queued: false,
            message_id: None,
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["queued"], false);
        assert!(json.get("messageId").is_none());
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
            prompt_collection_id: None,
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
            prompt_collection_id: None,
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
            input: "summarize yesterday".into(),
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
    fn transcript_entry_roundtrip() {
        let msg = TranscriptEntry {
            id: "m1".into(),
            message: TranscriptMessage::new(MessageRole::Assistant, "hello"),
            kind: TranscriptEntryKind::AssistantResponse,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: TranscriptEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn prompt_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::PromptDocumentCreate).unwrap(),
            "\"prompt.document.create\""
        );
        assert_eq!(
            serde_json::to_string(&Method::PromptCollectionCreate).unwrap(),
            "\"prompt.collection.create\""
        );
    }

    #[test]
    fn prompt_document_create_roundtrip() {
        let params = PromptDocumentCreateParams {
            id: "identity.md".into(),
            content: "# Identity\nI am helpful.".into(),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: PromptDocumentCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn prompt_collection_create_roundtrip() {
        let params = PromptCollectionCreateParams {
            id: "default".into(),
            name: "my collection".into(),
            description: Some("desc".into()),
            documents: vec!["identity.md".into(), "skills.md".into()],
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: PromptCollectionCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }
}
