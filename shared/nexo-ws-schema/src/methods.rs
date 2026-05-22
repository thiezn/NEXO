#[cfg(test)]
use nexo_spec::tool::ToolExecutionConstraints;
use nexo_spec::{model::LoadedModelInfo, tool::ToolSpec};
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
    #[serde(rename = "agent.stop")]
    AgentStop,
    #[serde(rename = "agent.context.append")]
    AgentContextAppend,
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
    /// Node → gateway: fetch a prefill payload by SHA.
    #[serde(rename = "prefill.fetch")]
    PrefillFetch,
    /// Client → gateway: create a markdown file.
    #[serde(rename = "prefill.markdown.create")]
    PrefillMarkdownCreate,
    /// Client → gateway: list markdown files.
    #[serde(rename = "prefill.markdown.list")]
    PrefillMarkdownList,
    /// Client → gateway: delete a markdown file.
    #[serde(rename = "prefill.markdown.delete")]
    PrefillMarkdownDelete,
    /// Client → gateway: create a prefill collection.
    #[serde(rename = "prefill.collection.create")]
    PrefillCollectionCreate,
    /// Client → gateway: list prefill collections.
    #[serde(rename = "prefill.collection.list")]
    PrefillCollectionList,
    /// Client → gateway: delete a prefill collection.
    #[serde(rename = "prefill.collection.delete")]
    PrefillCollectionDelete,
    /// Client → gateway → node: analyze an image using the model's vision capabilities.
    #[serde(rename = "image.analyze")]
    ImageAnalyze,
}

/// Agent run status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
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
    /// The model ID to use for inference. If omitted, any available LLM node is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Enable thinking mode (Gemma 4). When true the model emits reasoning
    /// tokens that are returned in the event but not persisted in history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
}

/// Parameters for the `agent.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentStopParams {
    /// The active run to stop.
    pub run_id: String,
}

/// Response payload for the `agent.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentStopResponse {
    /// Whether the run was still active and has now been stopped.
    pub stopped: bool,
}

/// Parameters for the `agent.context.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentContextAppendParams {
    /// The active run that should observe the new context on its next round.
    pub run_id: String,
    /// Arbitrary structured context to append to the transcript.
    pub context: serde_json::Value,
}

/// Response payload for the `agent.context.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentContextAppendResponse {
    /// Whether the context was accepted for the active run.
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

/// A single transcript message forwarded in a typed agent round request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentRoundMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Gateway-to-node payload for a single agent round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentRoundRequest {
    pub run_id: String,
    pub round_id: String,
    pub session_id: String,
    pub messages: Vec<AgentRoundMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpecEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

/// A single tool call returned from a node for a round.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentRoundToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Node-to-gateway response payload for a single agent round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentRoundResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<AgentRoundToolCall>,
}

/// A single tool entry in the tools catalog response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct SessionCreateParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// ID of a prefill collection to associate with this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_collection_id: Option<String>,
}

/// Response payload for `session.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateResponse {
    pub session_id: String,
    /// The prefill collection ID associated with this session, if one was provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_collection_id: Option<String>,
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

// -- model.load --

/// Parameters for `model.load` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadParams {
    pub model_id: String,
}

/// Response payload for `model.load`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadResponse {
    pub model_id: String,
    pub loaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// -- model.unload --

/// Parameters for `model.unload` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct ModelStatusParams {
    /// Models currently loaded with their categories.
    #[serde(default)]
    pub loaded_models: Vec<LoadedModelInfo>,
    /// All model IDs available on disk on this node.
    #[serde(default)]
    pub available_models: Vec<String>,
}

// -- prefill.fetch --

/// Parameters for `prefill.fetch` (node → gateway).
/// The node sends the SHA-256 hex of the combined collection content.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PrefillFetchParams {
    pub prefill_sha: String,
}

/// Response payload for `prefill.fetch`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PrefillFetchResponse {
    pub prefill_sha: String,
    pub content: String,
}

// -- prefill.markdown.create --

/// Parameters for `prefill.markdown.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownCreateParams {
    /// Filename for the markdown file (e.g. "identity.md").
    pub filename: String,
    pub content: String,
}

/// Response payload for `prefill.markdown.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownCreateResponse {
    pub filename: String,
}

// -- prefill.markdown.list --

/// Parameters for `prefill.markdown.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownListParams {}

/// A single markdown file entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct MarkdownFileEntry {
    pub filename: String,
}

/// Response payload for `prefill.markdown.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownListResponse {
    pub files: Vec<MarkdownFileEntry>,
}

// -- prefill.markdown.delete --

/// Parameters for `prefill.markdown.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownDeleteParams {
    pub filename: String,
}

/// Response payload for `prefill.markdown.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillMarkdownDeleteResponse {
    pub deleted: bool,
}

// -- prefill.collection.create --

/// Parameters for `prefill.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PrefillCollectionCreateParams {
    /// Unique ID for the collection.
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered list of markdown filenames that form this collection.
    pub markdown_files: Vec<String>,
}

/// Response payload for `prefill.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillCollectionCreateResponse {
    pub id: String,
}

// -- prefill.collection.list --

/// Parameters for `prefill.collection.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillCollectionListParams {}

/// A single prefill collection entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CollectionEntry {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered list of markdown filenames in this collection.
    pub markdown_files: Vec<String>,
}

/// Response payload for `prefill.collection.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillCollectionListResponse {
    pub collections: Vec<CollectionEntry>,
}

// -- prefill.collection.delete --

/// Parameters for `prefill.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillCollectionDeleteParams {
    pub id: String,
}

/// Response payload for `prefill.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PrefillCollectionDeleteResponse {
    pub deleted: bool,
}

// -- image.analyze --

/// Parameters for `image.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
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
    fn agent_params_roundtrip() {
        let params = AgentParams {
            prompt: "summarize this".into(),
            idempotency_key: "idem-1".into(),
            session_id: Some("sess-1".into()),
            context: Some(serde_json::json!({"files": ["a.rs"]})),
            model_id: None,
            thinking: None,
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
            model_id: None,
            thinking: None,
        };
        let json = serde_json::to_value(&params).unwrap();
        assert!(!json.as_object().unwrap().contains_key("sessionId"));
    }

    #[test]
    fn agent_control_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::AgentStop).unwrap(),
            "\"agent.stop\""
        );
        assert_eq!(
            serde_json::to_string(&Method::AgentContextAppend).unwrap(),
            "\"agent.context.append\""
        );
    }

    #[test]
    fn agent_stop_params_roundtrip() {
        let params = AgentStopParams {
            run_id: "run-1".into(),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: AgentStopParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn agent_context_append_params_roundtrip() {
        let params = AgentContextAppendParams {
            run_id: "run-1".into(),
            context: serde_json::json!({"files": ["notes.md"]}),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: AgentContextAppendParams = serde_json::from_str(&json).unwrap();
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
    fn agent_status_serialization() {
        for (status, expected) in [
            (AgentStatus::Accepted, "\"accepted\""),
            (AgentStatus::Queued, "\"queued\""),
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
    fn agent_round_request_roundtrip() {
        let request = AgentRoundRequest {
            run_id: "run-1".into(),
            round_id: "round-1".into(),
            session_id: "sess-1".into(),
            messages: vec![AgentRoundMessage {
                role: "system".into(),
                content: "You are helpful".into(),
                tool_call_id: None,
                tool_name: None,
            }],
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
        let decoded: AgentRoundRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request, decoded);
    }

    #[test]
    fn agent_round_response_roundtrip() {
        let response = AgentRoundResponse {
            content: Some("Final answer".into()),
            rationale: Some("Reasoning summary".into()),
            tool_calls: vec![AgentRoundToolCall {
                id: "call-1".into(),
                name: "echo.run".into(),
                arguments: serde_json::json!({"input": "hello"}),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        let decoded: AgentRoundResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response, decoded);
    }

    #[test]
    fn agent_context_append_response_omits_missing_message_id() {
        let response = AgentContextAppendResponse {
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
            prefill_collection_id: None,
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

    #[test]
    fn prefill_method_serialization() {
        assert_eq!(
            serde_json::to_string(&Method::PrefillFetch).unwrap(),
            "\"prefill.fetch\""
        );
        assert_eq!(
            serde_json::to_string(&Method::PrefillMarkdownCreate).unwrap(),
            "\"prefill.markdown.create\""
        );
        assert_eq!(
            serde_json::to_string(&Method::PrefillCollectionCreate).unwrap(),
            "\"prefill.collection.create\""
        );
    }

    #[test]
    fn prefill_markdown_create_roundtrip() {
        let params = PrefillMarkdownCreateParams {
            filename: "identity.md".into(),
            content: "# Identity\nI am helpful.".into(),
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: PrefillMarkdownCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }

    #[test]
    fn prefill_collection_create_roundtrip() {
        let params = PrefillCollectionCreateParams {
            id: "default".into(),
            name: "my collection".into(),
            description: Some("desc".into()),
            markdown_files: vec!["identity.md".into(), "skills.md".into()],
        };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: PrefillCollectionCreateParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, decoded);
    }
}
