use crate::{ModelCapability, ModelId, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ClientInfo, DeviceInfo, ProtocolInfo};

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
