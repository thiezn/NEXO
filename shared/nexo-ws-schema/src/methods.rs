use nexo_core::message::ConversationMessage;
use nexo_core::{
    ModelDescriptor, ReasoningSettings, SpeechLanguage, ToolCall, ToolChoice, ToolDefinition,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Available request methods in the gateway protocol.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    /// Client or node connect handshake.
    Connect,
    /// Health check request.
    Health,
    /// Gateway status request.
    Status,
    /// Generic send request.
    Send,
    #[serde(rename = "run.start")]
    /// Run start request.
    RunStart,
    #[serde(rename = "run.stop")]
    /// Run stop request.
    RunStop,
    #[serde(rename = "run.instructions.append")]
    /// Append instructions to an active run.
    RunInstructionsAppend,
    #[serde(rename = "run.round")]
    /// Execute one run round.
    RunRound,
    /// Presence update.
    SystemPresence,
    #[serde(rename = "tools.catalog")]
    /// Tool catalog request.
    ToolsCatalog,
    #[serde(rename = "tools.register")]
    /// Tool registration request.
    ToolsRegister,
    #[serde(rename = "tools.execute")]
    /// Tool execution request.
    ToolsExecute,
    #[serde(rename = "session.create")]
    /// Session create request.
    SessionCreate,
    #[serde(rename = "session.list")]
    /// Session list request.
    SessionList,
    #[serde(rename = "session.get")]
    /// Session get request.
    SessionGet,
    #[serde(rename = "session.clear")]
    /// Session clear request.
    SessionClear,
    #[serde(rename = "cron.create")]
    /// Cron create request.
    CronCreate,
    #[serde(rename = "cron.list")]
    /// Cron list request.
    CronList,
    #[serde(rename = "cron.delete")]
    /// Cron delete request.
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
    /// Client → gateway → node: analyze an audio clip using the model's audio capabilities.
    #[serde(rename = "audio.analyze")]
    AudioAnalyze,
    /// Client → gateway → node: generate one or more images from a text prompt.
    #[serde(rename = "image.generate")]
    ImageGenerate,
    /// Client → gateway → node: generate audio from a text prompt.
    #[serde(rename = "audio.generate")]
    AudioGenerate,
}

/// Run lifecycle status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Run accepted by the gateway.
    Accepted,
    /// Run is queued, waiting for an LLM node to become available.
    Queued,
    /// Run is building reasoning/thinking tokens.
    Thinking,
    /// Run is waiting for tool execution.
    ToolCall,
    /// Run is streaming output tokens.
    Streaming,
    /// Run completed successfully.
    Completed,
    /// Run failed.
    Failed,
    /// Run was cancelled.
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
    /// Field value.
    pub target: String,
    /// Field value.
    pub payload: serde_json::Value,
    /// Field value.
    pub idempotency_key: String,
}

/// Parameters for the `run.start` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunStartParams {
    /// Field value.
    pub input: String,
    /// Field value.
    pub idempotency_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub instructions: Option<serde_json::Value>,
    /// The model ID to use for inference. If omitted, any available LLM node is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub model_id: Option<String>,
    /// Optional reasoning controls for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub reasoning: Option<ReasoningSettings>,
    /// Optional provider-agnostic tool-use policy for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Parameters for the `run.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunStopParams {
    /// The active run to stop.
    pub run_id: String,
}

/// Response payload for the `run.stop` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunStopResponse {
    /// Whether the run was still active and has now been stopped.
    pub stopped: bool,
}

/// Parameters for the `run.instructions.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunInstructionsAppendParams {
    /// The active run that should observe the new context on its next round.
    pub run_id: String,
    /// Arbitrary structured instructions to append to the conversation.
    pub instructions: serde_json::Value,
}

/// Response payload for the `run.instructions.append` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunInstructionsAppendResponse {
    /// Whether the instructions were accepted for the active run.
    pub queued: bool,
    /// The persisted message identifier, when context was queued successfully.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub message_id: Option<String>,
}

/// Parameters for the `system-presence` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SystemPresenceParams {
    /// Field value.
    pub status: String,
}

/// Parameters for the `tools.catalog` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsCatalogParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub filter: Option<String>,
}

// -- Response payload types --

/// Response payload for `health`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    /// Field value.
    pub status: String,
    /// Field value.
    pub uptime_secs: u64,
}

/// Response payload for `status`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StatusResponse {
    /// Field value.
    pub connected_users: u32,
    /// Field value.
    pub connected_nodes: u32,
    /// Field value.
    pub capabilities: Vec<String>,
}

/// Response payload for `send`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SendResponse {
    /// Field value.
    pub delivered: bool,
}

/// Response payload for `run.start` (initial ack and final result).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunStartResponse {
    /// Field value.
    pub run_id: String,
    /// Field value.
    pub session_id: String,
    /// Field value.
    pub status: RunStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub summary: Option<String>,
}

/// Gateway-to-node payload for a single run round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunRoundRequest {
    /// Field value.
    pub run_id: String,
    /// Field value.
    pub round_id: String,
    /// Field value.
    pub session_id: String,
    /// Field value.
    pub messages: Vec<ConversationMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// Field value.
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    /// Tool use policy for the round.
    pub tool_choice: ToolChoice,
    #[serde(default)]
    /// Reasoning controls for the round.
    pub reasoning: ReasoningSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub model_id: Option<String>,
}

/// A single tool call returned from a node for a round.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunRoundToolCall {
    #[serde(flatten)]
    /// Field value.
    pub call: ToolCall,
}

/// Node-to-gateway response payload for a single run round inference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunRoundResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// Field value.
    pub tool_calls: Vec<RunRoundToolCall>,
}

/// A single tool entry in the tools catalog response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
pub struct ToolEntry {
    #[serde(flatten)]
    /// Field value.
    pub spec: ToolDefinition,
    /// Field value.
    pub source: String,
    /// Field value.
    pub available: bool,
}

impl ToolEntry {
    /// Create a tool catalog entry from a shared tool spec.
    pub fn new(spec: ToolDefinition, source: impl Into<String>, available: bool) -> Self {
        Self {
            spec,
            source: source.into(),
            available,
        }
    }
}

/// Response payload for `tools.catalog`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ToolsCatalogResponse {
    /// Field value.
    pub tools: Vec<ToolEntry>,
}

// -- tools.register --

/// A tool specification entry for registration.

/// Parameters for the `tools.register` method (sent by nodes).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ToolsRegisterParams {
    /// Field value.
    pub tools: Vec<ToolDefinition>,
}

/// Response payload for `tools.register`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolsRegisterResponse {
    /// Field value.
    pub registered: u32,
}

// -- tools.execute --

/// Parameters for the `tools.execute` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolsExecuteParams {
    /// Field value.
    pub tool: String,
    /// Field value.
    pub args: serde_json::Value,
    /// Field value.
    pub idempotency_key: String,
}

/// Response payload for `tools.execute`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ToolsExecuteResponse {
    /// Field value.
    pub success: bool,
    /// Field value.
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub error: Option<String>,
}

// -- session.create --

/// Parameters for the `session.create` method.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub name: Option<String>,
    /// ID of a prompt collection to associate with this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub prompt_collection_id: Option<String>,
}

/// Response payload for `session.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateResponse {
    /// Field value.
    pub session_id: String,
    /// The prompt collection ID associated with this session, if one was provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub prompt_collection_id: Option<String>,
}

// -- session.list --

/// Parameters for the `session.list` method (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionListParams {}

/// A single session entry in a session list response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntry {
    /// Field value.
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub prompt_collection_id: Option<String>,
    /// Field value.
    pub created_at: String,
    /// Field value.
    pub last_active_at: String,
    /// Field value.
    pub message_count: u32,
}

/// Response payload for `session.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionListResponse {
    /// Field value.
    pub sessions: Vec<SessionEntry>,
}

// -- session.get --

/// Parameters for the `session.get` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionGetParams {
    /// Field value.
    pub session_id: String,
}

/// Response payload for `session.get`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SessionGetResponse {
    /// Field value.
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub prompt_collection_id: Option<String>,
    /// Field value.
    pub messages: Vec<ConversationMessage>,
    /// Field value.
    pub created_at: String,
}

// -- session.clear --

/// Parameters for the `session.clear` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionClearParams {
    /// Field value.
    pub session_id: String,
}

/// Response payload for `session.clear`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionClearResponse {
    /// Field value.
    pub cleared: bool,
}

// -- cron.create --

/// Parameters for the `cron.create` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronCreateParams {
    /// Field value.
    pub name: String,
    /// Field value.
    pub schedule: String,
    /// Field value.
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub session_id: Option<String>,
}

/// Response payload for `cron.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronCreateResponse {
    /// Field value.
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
    /// Field value.
    pub job_id: String,
    /// Field value.
    pub name: String,
    /// Field value.
    pub schedule: String,
    /// Field value.
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub last_run_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Field value.
    pub next_run_at: Option<String>,
}

/// Response payload for `cron.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronListResponse {
    /// Field value.
    pub jobs: Vec<CronEntry>,
}

// -- cron.delete --

/// Parameters for the `cron.delete` method.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CronDeleteParams {
    /// Field value.
    pub job_id: String,
}

/// Response payload for `cron.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CronDeleteResponse {
    /// Field value.
    pub deleted: bool,
}

// -- model.load --

/// Parameters for `model.load` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadParams {
    /// Stable ID for the model to load.
    pub model_id: String,
}

/// Response payload for `model.load`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadResponse {
    /// Stable ID for the model.
    pub model_id: String,
    /// Indicates whether the model was successfully loaded.
    pub loaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Error message if the model failed to load.
    pub error: Option<String>,
}

// -- model.unload --

/// Parameters for `model.unload` (gateway → node).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelUnloadParams {
    /// Stable ID for the model to unload.
    pub model_id: String,
}

/// Response payload for `model.unload`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ModelUnloadResponse {
    /// Indicates whether the model was successfully unloaded.
    pub unloaded: bool,
}

// -- model.status --

/// Sent by a node to report its model state.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatusParams {
    /// Models currently loaded with their categories.
    #[serde(default)]
    pub loaded_models: Vec<ModelDescriptor>,
    /// All model IDs available on disk on this node.
    #[serde(default)]
    pub available_models: Vec<String>,
    /// All model descriptors available on disk on this node.
    #[serde(default)]
    pub available_model_descriptors: Vec<ModelDescriptor>,
}

// -- prompt.document.create --

/// A stored prompt document.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocument {
    /// Stable prompt document identifier (for example `identity.md`).
    pub id: String,
    /// Markdown or plain text content of the prompt document.
    pub content: String,
}

/// An ordered collection of prompt documents.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollection {
    /// Unique collection identifier.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Optional description shown in UIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered prompt document IDs that compose the collection.
    #[serde(default)]
    pub documents: Vec<String>,
}

/// The assembled system prompt sent to a model round.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SystemPrompt {
    /// Fully assembled prompt content passed to an inference round.
    pub content: String,
}

/// Parameters for `prompt.document.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentCreateParams {
    /// Stable ID for the prompt document (for example `identity.md`).
    pub id: String,
    /// Markdown or plain text content of the prompt document.
    pub content: String,
}

/// Response payload for `prompt.document.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentCreateResponse {
    /// Stable ID for the prompt document (for example `identity.md`).
    pub id: String,
}

// -- prompt.document.list --

/// Parameters for `prompt.document.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentListParams {}

/// A single prompt document entry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentEntry {
    /// Stable ID for the prompt document (for example `identity.md`).
    pub id: String,
}

/// Response payload for `prompt.document.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentListResponse {
    /// List of prompt document entries.
    pub documents: Vec<PromptDocumentEntry>,
}

// -- prompt.document.delete --

/// Parameters for `prompt.document.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentDeleteParams {
    /// Stable ID for the prompt document (for example `identity.md`).
    pub id: String,
}

/// Response payload for `prompt.document.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptDocumentDeleteResponse {
    /// Indicates whether the prompt document was successfully deleted.
    pub deleted: bool,
}

// -- prompt.collection.create --

/// Parameters for `prompt.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PromptCollectionCreateParams {
    /// Unique ID for the collection.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional description shown in UIs.
    pub description: Option<String>,
    /// Ordered list of prompt document IDs that form this collection.
    pub documents: Vec<String>,
}

/// Response payload for `prompt.collection.create`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionCreateResponse {
    /// Stable ID for the collection.
    pub id: String,
}

// -- prompt.collection.list --

/// Parameters for `prompt.collection.list` (empty).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionListParams {}

/// Response payload for `prompt.collection.list`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionListResponse {
    /// List of prompt collections.
    pub collections: Vec<PromptCollection>,
}

// -- prompt.collection.delete --

/// Parameters for `prompt.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionDeleteParams {
    /// Stable ID for the collection.
    pub id: String,
}

/// Response payload for `prompt.collection.delete`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptCollectionDeleteResponse {
    /// Indicates whether the collection was successfully deleted.
    pub deleted: bool,
}

// -- image.analyze --

/// Parameters for `image.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnalyzeParams {
    /// Base64-encoded image data.
    pub image_data: String,
    /// Optional session identifier used to preserve runtime continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional media type for the encoded image, such as `image/png`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// The prompt/question about the image.
    pub prompt: String,
    #[serde(default = "default_image_analyze_max_tokens")]
    /// Field value.
    pub max_tokens: usize,
    #[serde(default = "default_image_analyze_temperature")]
    /// Field value.
    pub temperature: f64,
    /// Visual token budget for variable resolution. Common values: 70, 140, 280, 560, 1120.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual_token_budget: Option<u32>,
    /// Unique key to ensure idempotency of the request.
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
    /// Textual analysis result of the image.
    pub text: String,
    /// Number of tokens generated during analysis.
    pub tokens_generated: usize,
    /// Time taken for inference in milliseconds.
    pub inference_time_ms: u64,
}

// -- audio.analyze --

/// Parameters for `audio.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioAnalyzeParams {
    /// Base64-encoded audio data.
    pub audio_data: String,
    /// Optional session identifier used to preserve runtime continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional media type for the encoded audio, such as `audio/wav`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Optional sample rate in hertz.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Optional channel count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_count: Option<u16>,
    /// The prompt/question about the audio clip.
    pub prompt: String,
    #[serde(default = "default_audio_analyze_max_tokens")]
    /// Field value.
    pub max_tokens: usize,
    #[serde(default = "default_audio_analyze_temperature")]
    /// Field value.
    pub temperature: f64,
    /// Unique key to ensure idempotency of the request.
    pub idempotency_key: String,
}

fn default_audio_analyze_max_tokens() -> usize {
    4096
}

fn default_audio_analyze_temperature() -> f64 {
    1.0
}

/// Response payload for `audio.analyze`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AudioAnalyzeResponse {
    /// Textual analysis result of the audio.
    pub text: String,
    /// Number of tokens generated during analysis.
    pub tokens_generated: usize,
    /// Time taken for inference in milliseconds.
    pub inference_time_ms: u64,
}

// -- image.generate --

/// Parameters for `image.generate`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerateParams {
    /// The positive prompt used for generation.
    pub prompt: String,
    /// Optional session identifier used to preserve runtime continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional negative prompt used to suppress unwanted attributes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Requested image width in pixels.
    #[serde(default = "default_image_generate_width")]
    pub width: u32,
    /// Requested image height in pixels.
    #[serde(default = "default_image_generate_height")]
    pub height: u32,
    /// Number of images to generate.
    #[serde(default = "default_image_generate_sample_count")]
    pub sample_count: u32,
    /// Optional diffusion step count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Optional guidance scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,
    /// Optional deterministic random seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Unique key to ensure idempotency of the request.
    pub idempotency_key: String,
}

fn default_image_generate_width() -> u32 {
    1024
}

fn default_image_generate_height() -> u32 {
    1024
}

fn default_image_generate_sample_count() -> u32 {
    1
}

/// A generated image returned by `image.generate`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedImagePayload {
    /// Zero-based index of this image in the generated batch.
    pub index: usize,
    /// Base64-encoded image bytes.
    pub image_data: String,
    /// Optional media type of the generated image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Generated image width, in pixels, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Generated image height, in pixels, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// Response payload for `image.generate`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerateResponse {
    /// Generated images.
    pub images: Vec<GeneratedImagePayload>,
    /// Time taken for inference in milliseconds.
    pub inference_time_ms: u64,
}

// -- audio.generate --

/// Parameters for `audio.generate`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioGenerateParams {
    /// The text prompt to synthesize into audio.
    pub prompt: String,
    /// Optional session identifier used to preserve runtime continuity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Requested spoken language.
    #[serde(default)]
    pub language: SpeechLanguage,
    /// Optional voice label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Optional output sample rate in hertz.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Optional speaking speed multiplier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Unique key to ensure idempotency of the request.
    pub idempotency_key: String,
}

/// Response payload for `audio.generate`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AudioGenerateResponse {
    /// Base64-encoded audio bytes.
    pub audio_data: String,
    /// Optional media type of the generated audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// The generated audio format.
    pub format: String,
    /// Audio sample rate, in hertz, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Number of audio channels, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_count: Option<u16>,
    /// Time taken for inference in milliseconds.
    pub inference_time_ms: u64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use nexo_core::ToolExecutionConstraints;
    use nexo_core::message::{ContentPart, MessageRole, TextPart};
    use std::collections::HashMap;

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
            reasoning: Some(nexo_core::ReasoningSettings {
                thinking: nexo_core::ThinkingMode::Enabled,
                effort: Some(nexo_core::ReasoningEffort::High),
            }),
            tool_choice: Some(ToolChoice::Specific {
                name: "echo.run".into(),
            }),
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
            reasoning: None,
            tool_choice: None,
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
                ToolDefinition {
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
            ToolDefinition {
                name: "echo".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: Some("2026-05-22".into()),
                execution: ToolExecutionConstraints {
                    side_effect_level: nexo_core::ToolSideEffectLevel::ReadOnly,
                    parallelism: nexo_core::ToolParallelism::ParallelGlobal,
                    timeout_ms: Some(5_000),
                },
            },
            "node",
            true,
        );
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["parameters"]["type"], "object");
        assert_eq!(json["contractVersion"], "2026-05-22");
        assert_eq!(json["execution"]["sideEffectLevel"], "read_only");
        assert_eq!(json["execution"]["parallelism"], "parallel_global");
        assert_eq!(json["execution"]["timeoutMs"], 5_000);
    }

    #[test]
    fn tool_entry_omits_default_spec_metadata() {
        let entry = ToolEntry::new(
            ToolDefinition {
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
        assert!(json.as_object().unwrap().contains_key("execution"));
        assert!(json.as_object().unwrap().contains_key("metadata"));
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
            tools: vec![ToolDefinition {
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
            messages: vec![ConversationMessage {
                role: MessageRole::System,
                parts: vec![ContentPart::Text(TextPart {
                    text: "You are helpful".into(),
                })],
                metadata: HashMap::new(),
            }],
            tools: vec![ToolDefinition {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: None,
                execution: ToolExecutionConstraints::default(),
            }],
            tool_choice: ToolChoice::Specific {
                name: "echo.run".into(),
            },
            reasoning: nexo_core::ReasoningSettings::default(),
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
                call: ToolCall {
                    id: "call-1".into(),
                    index: 0,
                    name: "echo.run".into(),
                    arguments: serde_json::json!({"input": "hello"}),
                },
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
    fn conversation_message_roundtrip() {
        let msg = ConversationMessage {
            role: MessageRole::Assistant,
            parts: vec![ContentPart::Text(TextPart {
                text: "hello".into(),
            })],
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: ConversationMessage = serde_json::from_str(&json).unwrap();
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
