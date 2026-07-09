use crate::{ModelCapability, ModelId, PeerId, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::{ClientInfo, DeviceInfo, ProtocolInfo};
/// A single active Node in the Nexo Gateway.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct Node {
    /// Unique identifier for this node, derived from stable client and device identifiers.
    id: PeerId,

    /// The current state of the node.
    state: NodeState,

    /// Tool definitions exposed by this node.
    #[serde(default)]
    tools: HashSet<ToolDefinition>,

    /// Model IDs available on disk for this node.
    #[serde(default)]
    models_on_disk: HashSet<ModelId>,

    /// All models currently loaded in memory for this node.
    #[serde(default)]
    models_in_memory: HashSet<ModelId>,

    /// Connected at
    connected_at: chrono::DateTime<chrono::Utc>,
}

impl Node {
    /// Initialize a new node with the given peer ID, state, tools, and models.
    pub fn new(
        id: PeerId,
        state: NodeState,
        tools: HashSet<ToolDefinition>,
        models_on_disk: HashSet<ModelId>,
        models_in_memory: HashSet<ModelId>,
    ) -> Self {
        let connected_at = chrono::Utc::now();
        Self {
            id,
            state,
            tools,
            models_on_disk,
            models_in_memory,
            connected_at,
        }
    }

    /// Build a node from the given node properties and state.
    pub fn from_properties(
        properties: &NodeProperties,
        state: NodeState,
        models_in_memory: HashSet<ModelId>,
    ) -> Self {
        let id = PeerId::new(properties.client().id, properties.device().id);
        let tools = properties.tools().iter().cloned().collect();
        let models_on_disk = properties.models().iter().cloned().collect();
        Self::new(id, state, tools, models_on_disk, models_in_memory)
    }

    /// Return the unique identifier for this node.
    pub fn id(&self) -> PeerId {
        self.id
    }

    /// Return the current state of this node.
    pub fn state(&self) -> &NodeState {
        &self.state
    }

    /// Return the tool definitions exposed by this node.
    pub fn tools(&self) -> &HashSet<ToolDefinition> {
        &self.tools
    }

    /// Return the model IDs available on disk for this node.
    pub fn models_on_disk(&self) -> &HashSet<ModelId> {
        &self.models_on_disk
    }

    /// Return the model IDs currently loaded in memory for this node.
    pub fn models_in_memory(&self) -> &HashSet<ModelId> {
        &self.models_in_memory
    }

    /// Get the timestamp when this node connected.
    pub fn connected_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.connected_at
    }
}

/// The current state of the Node.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub enum NodeState {
    /// The node is idle and not currently processing any requests.
    Idle,

    /// Loading a model into memory for inference or tool calling.
    LoadingModel,

    /// Unloading a model from memory to free up resources.
    UnloadingModel,

    /// The node is currently processing an inference request.
    RunningInference,

    /// The node is currently processing a tool call request.
    RunningToolCall,
}

/// Persisted configuration and advertised runtime state for a Nexo node.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct NodeProperties {
    /// WebSocket URL of the gateway this node connects to.
    gateway_url: String,

    /// Shared gateway authentication token.
    auth_token: String,

    /// Delay between reconnect attempts after a gateway disconnect.
    reconnect_interval_ms: u64,

    /// Optional proxy URL used by local model download commands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proxy: Option<String>,

    /// Capabilities that should be covered by loaded models at startup.
    #[serde(default)]
    startup_capabilities: Vec<ModelCapability>,

    /// Default model for each startup capability. Missing values use catalog selection.
    #[serde(default)]
    default_models: HashMap<ModelCapability, ModelId>,

    /// Protocol negotiation metadata sent during connect.
    protocol: ProtocolInfo,

    /// Client identity metadata for this node.
    #[serde(default)]
    client: ClientInfo,

    /// Device identity used for pairing.
    #[serde(default)]
    device: DeviceInfo,

    /// Tool definitions exposed by this node.
    #[serde(default)]
    tools: Vec<ToolDefinition>,

    /// Model IDs available on disk for this node.
    #[serde(default)]
    models: Vec<ModelId>,
}

impl NodeProperties {
    /// Start building node properties with explicit identity and auth.
    pub fn builder(
        client: ClientInfo,
        device: DeviceInfo,
        auth_token: impl Into<String>,
    ) -> NodePropertiesBuilder {
        NodePropertiesBuilder::new(client, device, auth_token)
    }

    /// Build node properties with default runtime settings.
    pub fn new(client: ClientInfo, device: DeviceInfo, auth_token: impl Into<String>) -> Self {
        Self::builder(client, device, auth_token).build()
    }

    /// Return a builder initialized from these properties.
    pub fn into_builder(self) -> NodePropertiesBuilder {
        NodePropertiesBuilder {
            gateway_url: self.gateway_url,
            auth_token: self.auth_token,
            reconnect_interval_ms: self.reconnect_interval_ms,
            proxy: self.proxy,
            startup_capabilities: self.startup_capabilities,
            default_models: self.default_models,
            client: self.client,
            device: self.device,
            tools: self.tools,
            models: self.models,
        }
    }

    /// WebSocket URL of the gateway this node connects to.
    pub fn gateway_url(&self) -> &str {
        &self.gateway_url
    }

    /// Shared gateway authentication token.
    pub fn auth_token(&self) -> &str {
        &self.auth_token
    }

    /// Delay between reconnect attempts after a gateway disconnect.
    pub fn reconnect_interval_ms(&self) -> u64 {
        self.reconnect_interval_ms
    }

    /// Optional proxy URL used by local model download commands.
    pub fn proxy(&self) -> Option<&str> {
        self.proxy.as_deref()
    }

    /// Capabilities that should be covered by loaded models at startup.
    pub fn startup_capabilities(&self) -> &[ModelCapability] {
        &self.startup_capabilities
    }

    /// Default model for each startup capability.
    pub fn default_models(&self) -> &HashMap<ModelCapability, ModelId> {
        &self.default_models
    }

    /// Protocol negotiation metadata sent during connect.
    pub fn protocol(&self) -> &ProtocolInfo {
        &self.protocol
    }

    /// Client identity metadata for this node.
    pub fn client(&self) -> &ClientInfo {
        &self.client
    }

    /// Device identity used for pairing.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Tool definitions exposed by this node.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Model IDs available on disk for this node.
    pub fn models(&self) -> &[ModelId] {
        &self.models
    }

    /// Return a set of ModelIds that are required to be loaded at startup,
    /// based on the startup capabilities and default models.
    pub fn startup_models(&self) -> HashSet<ModelId> {
        let mut startup_models = HashSet::new();
        for capability in &self.startup_capabilities {
            if let Some(model_id) = self.default_models.get(capability) {
                startup_models.insert(model_id.clone());
            }
        }
        startup_models
    }
}

impl Default for NodeProperties {
    fn default() -> Self {
        Self::new(ClientInfo::default(), DeviceInfo::default(), "")
    }
}

/// Builder for [`NodeProperties`].
#[derive(Debug, Clone)]
pub struct NodePropertiesBuilder {
    gateway_url: String,
    auth_token: String,
    reconnect_interval_ms: u64,
    proxy: Option<String>,
    startup_capabilities: Vec<ModelCapability>,
    default_models: HashMap<ModelCapability, ModelId>,
    client: ClientInfo,
    device: DeviceInfo,
    tools: Vec<ToolDefinition>,
    models: Vec<ModelId>,
}

impl NodePropertiesBuilder {
    /// Create a node properties builder with required identity and auth values.
    pub fn new(client: ClientInfo, device: DeviceInfo, auth_token: impl Into<String>) -> Self {
        Self {
            gateway_url: "ws://127.0.0.1:6969".to_string(),
            auth_token: auth_token.into(),
            reconnect_interval_ms: 5000,
            proxy: None,
            startup_capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::ToolCalling,
            ],
            default_models: HashMap::new(),
            client,
            device,
            tools: Vec::new(),
            models: Vec::new(),
        }
    }

    /// Set the gateway URL.
    pub fn gateway_url(mut self, gateway_url: impl Into<String>) -> Self {
        self.gateway_url = gateway_url.into();
        self
    }

    /// Set the reconnect interval in milliseconds.
    pub fn reconnect_interval_ms(mut self, reconnect_interval_ms: u64) -> Self {
        self.reconnect_interval_ms = reconnect_interval_ms;
        self
    }

    /// Set the optional proxy URL used by local model download commands.
    pub fn proxy(mut self, proxy: Option<impl Into<String>>) -> Self {
        self.proxy = proxy.map(Into::into);
        self
    }

    /// Set startup capabilities.
    pub fn startup_capabilities(mut self, startup_capabilities: Vec<ModelCapability>) -> Self {
        self.startup_capabilities = startup_capabilities;
        self
    }

    /// Set default models by capability.
    pub fn default_models(mut self, default_models: HashMap<ModelCapability, ModelId>) -> Self {
        self.default_models = default_models;
        self
    }

    /// Set tool definitions exposed by this node.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set model IDs available on disk for this node.
    pub fn models(mut self, models: Vec<ModelId>) -> Self {
        self.models = models;
        self
    }

    /// Build complete node properties.
    pub fn build(self) -> NodeProperties {
        let protocol = ProtocolInfo::new_node(&self.client);
        NodeProperties {
            gateway_url: self.gateway_url,
            auth_token: self.auth_token,
            reconnect_interval_ms: self.reconnect_interval_ms,
            proxy: self.proxy,
            startup_capabilities: self.startup_capabilities,
            default_models: self.default_models,
            protocol,
            client: self.client,
            device: self.device,
            tools: self.tools,
            models: self.models,
        }
    }
}
