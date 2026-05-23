use crate::memory::git::GitStorage;
use crate::tools::GatewayToolExecutor;
use nexo_spec::model::{LoadedModelInfo, ModelCategory};
use nexo_ws_schema::{Frame, Role, Scope, ToolEntry, ToolSpecEntry};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Notify, RwLock, broadcast, mpsc, oneshot};

/// Unique identifier assigned to a connected peer.
pub type PeerId = String;

/// Information about a connected peer (user or node).
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Gateway-generated identifier for this connection.
    pub id: PeerId,
    /// Stable client identifier provided by the connecting peer.
    pub client_id: String,
    /// Peer role used for routing and authorization decisions.
    pub role: Role,
    /// Declared protocol scopes granted to the peer.
    pub scopes: Vec<Scope>,
    /// Declared capability families available on the peer.
    pub capabilities: Vec<String>,
    /// Declared command names supported by the peer.
    pub commands: Vec<String>,
    /// Optional persisted device identifier attached during connect.
    pub device_id: Option<String>,
    /// Timestamp at which the peer connected to the gateway.
    pub connected_at: chrono::DateTime<chrono::Utc>,
}

/// A tool registered by a node.
#[derive(Debug, Clone)]
pub struct RegisteredTool {
    /// Tool specification advertised by the node.
    pub spec: ToolSpecEntry,
    /// Peer currently hosting the tool.
    pub peer_id: PeerId,
    /// Timestamp at which the tool was registered.
    pub registered_at: chrono::DateTime<chrono::Utc>,
}

/// Shared mutable state for the gateway.
pub struct GatewayState {
    /// Connected peers keyed by gateway-assigned peer ID.
    pub peers: HashMap<PeerId, PeerInfo>,
    /// Directed senders used to push frames to a specific peer.
    pub peer_senders: HashMap<PeerId, mpsc::Sender<Frame>>,
    /// Node-hosted tool registrations keyed by tool name.
    pub tool_registry: HashMap<String, RegisteredTool>,
    /// Pending forwarded requests waiting for a response frame.
    pub pending_requests: HashMap<String, oneshot::Sender<Frame>>,
    /// Broadcast channel used for shared event fan-out.
    pub event_tx: broadcast::Sender<Frame>,
    /// Timestamp at which the gateway state was initialized.
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Models currently loaded in VRAM per node with their categories.
    pub loaded_models: HashMap<PeerId, Vec<LoadedModelInfo>>,
    /// Model IDs available on disk per node (declared at connect time).
    pub available_models: HashMap<PeerId, Vec<String>>,
    /// Notified whenever a node's loaded model changes (used to wake the queue drain watcher).
    pub model_ready_notify: Arc<Notify>,
    /// Resolved path to the storage root (~/.nexo/storage).
    pub storage_root: PathBuf,
    /// Tools that execute locally on the gateway (e.g., notes).
    pub gateway_tools: GatewayToolExecutor,
    /// Git-backed storage for persistent data (notes, prefill, SOUL.md).
    pub git_storage: Option<Arc<GitStorage>>,
}

impl GatewayState {
    /// Create a new empty gateway state rooted at the provided storage path.
    pub fn new(storage_root: PathBuf) -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            peers: HashMap::new(),
            peer_senders: HashMap::new(),
            tool_registry: HashMap::new(),
            pending_requests: HashMap::new(),
            event_tx: tx,
            started_at: chrono::Utc::now(),
            loaded_models: HashMap::new(),
            available_models: HashMap::new(),
            model_ready_notify: Arc::new(Notify::new()),
            storage_root,
            gateway_tools: GatewayToolExecutor::new(),
            git_storage: None,
        }
    }

    fn find_node_peer(
        &self,
        mut matches: impl FnMut(&PeerId, &PeerInfo) -> bool,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.peers.iter().find_map(|(peer_id, peer)| {
            if peer.role != Role::Node || !matches(peer_id, peer) {
                return None;
            }

            self.peer_senders
                .get(peer_id)
                .cloned()
                .map(|sender| (peer_id.clone(), sender))
        })
    }

    /// Register a newly connected peer and its directed sender channel.
    pub fn add_peer(&mut self, info: PeerInfo, sender: mpsc::Sender<Frame>) {
        tracing::info!(
            "Peer connected: {} (role={:?}, client={})",
            info.id,
            info.role,
            info.client_id
        );
        self.peer_senders.insert(info.id.clone(), sender);
        self.peers.insert(info.id.clone(), info);
    }

    /// Remove a disconnected peer and any state derived from its connection.
    pub fn remove_peer(&mut self, id: &str) {
        if let Some(peer) = self.peers.remove(id) {
            tracing::info!("Peer disconnected: {} (client={})", peer.id, peer.client_id);
        }
        self.peer_senders.remove(id);
        self.deregister_tools_for_peer(id);
        self.loaded_models.remove(id);
        self.available_models.remove(id);
    }

    /// Update the set of models available on disk for a peer.
    pub fn set_available_models(&mut self, peer_id: &str, models: Vec<String>) {
        self.available_models.insert(peer_id.to_string(), models);
    }

    /// Update the loaded models for a node. Notifies queue drain waiters.
    pub fn set_loaded_models(&mut self, peer_id: &str, models: Vec<LoadedModelInfo>) {
        self.loaded_models.insert(peer_id.to_string(), models);
        self.model_ready_notify.notify_waiters();
    }

    /// Find the first node that has `model_id` loaded in VRAM.
    pub fn find_loaded_llm_peer(&self, model_id: &str) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.find_node_peer(|peer_id, _| {
            self.loaded_models
                .get(peer_id)
                .is_some_and(|models| models.iter().any(|model| model.model_id == model_id))
        })
    }

    /// Find the first node that has `model_id` available on disk (not necessarily loaded).
    pub fn find_capable_peer_for_model(
        &self,
        model_id: &str,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.find_node_peer(|peer_id, _| {
            self.available_models.get(peer_id).is_some_and(|models| {
                models
                    .iter()
                    .any(|available_model| available_model == model_id)
            })
        })
    }

    /// Find the first node that has a loaded model matching the given category predicate.
    fn find_node_with_loaded_category(
        &self,
        pred: impl Fn(&ModelCategory) -> bool,
    ) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.find_node_peer(|peer_id, _| {
            self.loaded_models.get(peer_id).is_some_and(|models| {
                models
                    .iter()
                    .any(|model| model.categories.iter().any(&pred))
            })
        })
    }

    /// Find any connected node with a Chat or Tool model loaded.
    pub fn find_any_llm_peer(&self) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.find_node_with_loaded_category(is_llm_category)
    }

    /// Find any connected node with an Image model loaded.
    pub fn find_image_analyze_peer(&self) -> Option<(PeerId, mpsc::Sender<Frame>)> {
        self.find_node_with_loaded_category(|c| matches!(c, ModelCategory::Image))
    }

    /// Find all connected user peers for a routing identity, excluding the origin peer.
    pub fn find_user_peers_by_client_id(
        &self,
        client_id: &str,
        exclude_peer_id: &str,
    ) -> Vec<(PeerId, mpsc::Sender<Frame>)> {
        self.peers
            .iter()
            .filter_map(|(peer_id, peer)| {
                if peer.role != Role::User
                    || peer.client_id != client_id
                    || peer_id == exclude_peer_id
                {
                    return None;
                }

                self.peer_senders
                    .get(peer_id)
                    .cloned()
                    .map(|sender| (peer_id.clone(), sender))
            })
            .collect()
    }

    /// Returns true if any node has a Chat or Tool model loaded.
    pub fn has_llm_peer(&self) -> bool {
        self.find_any_llm_peer().is_some()
    }

    /// Register tools provided by a node. Returns the number of tools registered.
    pub fn register_tools(&mut self, peer_id: &str, tools: Vec<ToolSpecEntry>) -> u32 {
        let count = tools.len() as u32;
        let now = chrono::Utc::now();
        for spec in tools {
            tracing::debug!("Registered tool '{}' from peer {peer_id}", spec.name,);
            self.tool_registry.insert(
                spec.name.clone(),
                RegisteredTool {
                    spec,
                    peer_id: peer_id.to_string(),
                    registered_at: now,
                },
            );
        }
        count
    }

    /// Remove all tools registered by a specific peer.
    pub fn deregister_tools_for_peer(&mut self, peer_id: &str) {
        let before = self.tool_registry.len();
        self.tool_registry.retain(|name, tool| {
            if tool.peer_id == peer_id {
                tracing::debug!("Deregistered tool '{name}' (peer {peer_id} disconnected)");
                false
            } else {
                true
            }
        });
        let removed = before - self.tool_registry.len();
        if removed > 0 {
            tracing::info!("Deregistered {removed} tool(s) for peer {peer_id}");
        }
    }

    /// Look up a registered tool by name.
    pub fn find_tool(&self, name: &str) -> Option<&RegisteredTool> {
        self.tool_registry.get(name)
    }

    /// Build tool catalog entries from the registry (node tools + gateway-native tools).
    pub fn all_tool_entries(&self) -> Vec<ToolEntry> {
        let mut entries: Vec<ToolEntry> = self
            .tool_registry
            .values()
            .map(|rt| {
                ToolEntry::new(
                    rt.spec.clone(),
                    "node",
                    self.peer_senders.contains_key(&rt.peer_id),
                )
            })
            .collect();
        entries.extend(self.gateway_tools.tool_entries());
        entries
    }

    /// Count currently connected user peers.
    pub fn connected_users(&self) -> u32 {
        self.peers.values().filter(|p| p.role == Role::User).count() as u32
    }

    /// Count currently connected node peers.
    pub fn connected_nodes(&self) -> u32 {
        self.peers.values().filter(|p| p.role == Role::Node).count() as u32
    }

    /// Return the deduplicated capability families exposed by connected nodes.
    pub fn all_capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = self
            .peers
            .values()
            .filter(|p| p.role == Role::Node)
            .flat_map(|peer| peer.capabilities.iter().cloned())
            .collect();
        caps.sort();
        caps.dedup();
        caps
    }

    /// Return the gateway uptime in whole seconds.
    pub fn uptime_secs(&self) -> u64 {
        (chrono::Utc::now() - self.started_at).num_seconds().max(0) as u64
    }

    /// Get the client_id (used as user_id) for a peer.
    pub fn user_id_for_peer(&self, peer_id: &str) -> Option<String> {
        self.peers.get(peer_id).map(|p| p.client_id.clone())
    }
}

/// Shared, asynchronously mutable gateway state handle.
pub type SharedState = Arc<RwLock<GatewayState>>;

fn is_llm_category(c: &ModelCategory) -> bool {
    matches!(c, ModelCategory::Chat | ModelCategory::Tool)
}

#[cfg(test)]
pub(crate) fn dummy_sender() -> mpsc::Sender<Frame> {
    let (tx, _rx) = mpsc::channel(1);
    tx
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn make_loaded_model(model_id: &str, categories: Vec<ModelCategory>) -> LoadedModelInfo {
        LoadedModelInfo {
            model_id: model_id.into(),
            categories,
        }
    }

    fn make_user_peer(id: &str) -> PeerInfo {
        PeerInfo {
            id: id.into(),
            client_id: "cli".into(),
            role: Role::User,
            scopes: vec![Scope::UserRead],
            capabilities: vec![],
            commands: vec![],
            device_id: Some("dev-1".into()),
            connected_at: chrono::Utc::now(),
        }
    }

    fn make_node_peer(id: &str, capabilities: Vec<String>) -> PeerInfo {
        PeerInfo {
            id: id.into(),
            client_id: "rust-node".into(),
            role: Role::Node,
            scopes: vec![],
            capabilities,
            commands: vec![],
            device_id: Some("dev-2".into()),
            connected_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn add_and_remove_peer() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_user_peer("peer-1"), dummy_sender());
        assert_eq!(state.peers.len(), 1);
        state.remove_peer("peer-1");
        assert!(state.peers.is_empty());
    }

    #[test]
    fn connected_users_and_nodes_count() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_user_peer("u1"), dummy_sender());
        state.add_peer(make_user_peer("u2"), dummy_sender());
        state.add_peer(make_node_peer("n1", vec!["epub".into()]), dummy_sender());
        assert_eq!(state.connected_users(), 2);
        assert_eq!(state.connected_nodes(), 1);
    }

    #[test]
    fn find_user_peers_by_client_id_excludes_origin() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(
            PeerInfo {
                id: "sender".into(),
                client_id: "user-a".into(),
                role: Role::User,
                scopes: vec![Scope::UserRead],
                capabilities: vec![],
                commands: vec![],
                device_id: Some("dev-1".into()),
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );
        state.add_peer(
            PeerInfo {
                id: "target".into(),
                client_id: "user-b".into(),
                role: Role::User,
                scopes: vec![Scope::UserRead],
                capabilities: vec![],
                commands: vec![],
                device_id: Some("dev-2".into()),
                connected_at: chrono::Utc::now(),
            },
            dummy_sender(),
        );

        let recipients = state.find_user_peers_by_client_id("user-b", "sender");
        assert_eq!(recipients.len(), 1);
        assert_eq!(recipients[0].0, "target");
    }

    #[test]
    fn all_capabilities_aggregation() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(
            make_node_peer("n1", vec!["epub".into(), "game".into()]),
            dummy_sender(),
        );
        state.add_peer(
            make_node_peer("n2", vec!["game".into(), "tts".into()]),
            dummy_sender(),
        );
        let caps = state.all_capabilities();
        assert_eq!(caps, vec!["epub", "game", "tts"]);
    }

    #[test]
    fn all_capabilities_empty_with_no_nodes() {
        let state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        assert!(state.all_capabilities().is_empty());
    }

    #[test]
    fn peer_selection_finds_loaded_model_peer() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_user_peer("u1"), dummy_sender());
        state.add_peer(make_node_peer("n1", vec![]), dummy_sender());
        state.add_peer(make_node_peer("n2", vec![]), dummy_sender());
        state.set_loaded_models(
            "n1",
            vec![make_loaded_model("gemma-3n", vec![ModelCategory::Chat])],
        );
        state.set_loaded_models(
            "n2",
            vec![make_loaded_model("qwen-image", vec![ModelCategory::Image])],
        );

        let (peer_id, _sender) = state.find_loaded_llm_peer("gemma-3n").unwrap();
        assert_eq!(peer_id, "n1");
        assert!(state.find_loaded_llm_peer("missing-model").is_none());
    }

    #[test]
    fn peer_selection_finds_available_model_peer() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_node_peer("n1", vec![]), dummy_sender());
        state.add_peer(make_node_peer("n2", vec![]), dummy_sender());
        state.set_available_models("n1", vec!["chat-a".into()]);
        state.set_available_models("n2", vec!["image-b".into(), "chat-b".into()]);

        let (peer_id, _sender) = state.find_capable_peer_for_model("chat-b").unwrap();
        assert_eq!(peer_id, "n2");
        assert!(state.find_capable_peer_for_model("unknown").is_none());
    }

    #[test]
    fn peer_selection_matches_loaded_categories() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_node_peer("n-chat", vec![]), dummy_sender());
        state.add_peer(make_node_peer("n-image", vec![]), dummy_sender());
        state.set_loaded_models(
            "n-chat",
            vec![make_loaded_model(
                "chatty",
                vec![ModelCategory::Tool, ModelCategory::Chat],
            )],
        );
        state.set_loaded_models(
            "n-image",
            vec![make_loaded_model("vision", vec![ModelCategory::Image])],
        );

        let (llm_peer_id, _sender) = state.find_any_llm_peer().unwrap();
        assert_eq!(llm_peer_id, "n-chat");

        let (image_peer_id, _sender) = state.find_image_analyze_peer().unwrap();
        assert_eq!(image_peer_id, "n-image");
        assert!(state.has_llm_peer());
    }

    #[test]
    fn uptime_is_non_negative() {
        let state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        assert!(state.uptime_secs() < 2);
    }

    #[test]
    fn register_and_deregister_tools() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_node_peer("n1", vec!["echo".into()]), dummy_sender());

        let tools = vec![
            ToolSpecEntry {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: None,
                execution: Default::default(),
            },
            ToolSpecEntry {
                name: "echo.ping".into(),
                description: "Ping".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: None,
                execution: Default::default(),
            },
        ];
        let count = state.register_tools("n1", tools);
        assert_eq!(count, 2);
        assert_eq!(state.tool_registry.len(), 2);
        assert!(state.find_tool("echo.run").is_some());
        assert!(state.find_tool("echo.ping").is_some());

        // Remove peer should deregister tools
        state.remove_peer("n1");
        assert!(state.tool_registry.is_empty());
    }

    #[test]
    fn all_tool_entries_builds_catalog() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        state.add_peer(make_node_peer("n1", vec!["echo".into()]), dummy_sender());
        state.register_tools(
            "n1",
            vec![ToolSpecEntry {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
                contract_version: Some("2026-05-22".into()),
                execution: Default::default(),
            }],
        );

        let entries = state.all_tool_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].spec.name, "echo.run");
        assert_eq!(entries[0].source, "node");
        assert!(entries[0].available);
        assert_eq!(entries[0].spec.parameters["type"], "object");
    }

    #[test]
    fn tool_from_disconnected_peer_shows_unavailable() {
        let mut state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        // Manually insert a tool without a matching peer sender
        state.tool_registry.insert(
            "orphan.tool".into(),
            RegisteredTool {
                spec: ToolSpecEntry {
                    name: "orphan.tool".into(),
                    description: "Orphan".into(),
                    parameters: serde_json::json!({}),
                    contract_version: None,
                    execution: Default::default(),
                },
                peer_id: "gone".into(),
                registered_at: chrono::Utc::now(),
            },
        );
        let entries = state.all_tool_entries();
        assert_eq!(entries.len(), 1);
        assert!(!entries[0].available);
    }
}
