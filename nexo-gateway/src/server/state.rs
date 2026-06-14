use crate::memory::git::GitStorage;
use nexo_core::{ModelDefinition, ToolRegistry};
use nexo_ws_schema::{ConnectionRole, Frame, Scope, ToolDefinition, ToolEntry};
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
    pub role: ConnectionRole,
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
    pub definition: ToolDefinition,
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
    pub node_tool_registry: HashMap<String, RegisteredTool>,
    /// Pending forwarded requests waiting for a response frame.
    pub pending_requests: HashMap<String, oneshot::Sender<Frame>>,
    /// Broadcast channel used for shared event fan-out.
    pub event_tx: broadcast::Sender<Frame>,
    /// Timestamp at which the gateway state was initialized.
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Models currently loaded in VRAM per node.
    pub loaded_models: HashMap<PeerId, Vec<ModelDefinition>>,
    /// Model IDs available on disk per node (declared at connect time).
    pub available_models: HashMap<PeerId, Vec<String>>,
    /// Model descriptors available on disk per node (reported by model.status).
    pub available_model_descriptors: HashMap<PeerId, Vec<ModelDefinition>>,
    /// Queued multimodal generation request counts keyed by session ID.
    pub queued_generation_by_session: HashMap<String, usize>,
    /// Notified whenever a node's loaded model changes (used to wake the queue drain watcher).
    pub model_ready_notify: Arc<Notify>,
    /// Resolved path to the storage root (~/.nexo/storage).
    pub storage_root: PathBuf,
    /// Tools that execute locally on the gateway (e.g., notes).
    pub gateway_tool_registry: Arc<ToolRegistry>,
    /// Git-backed storage for persistent data such as notes and prompt documents.
    pub git_storage: Option<Arc<GitStorage>>,
}

impl GatewayState {
    /// Create a new empty gateway state rooted at the provided storage path.
    pub fn new(storage_root: PathBuf) -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            peers: HashMap::new(),
            peer_senders: HashMap::new(),
            node_tool_registry: HashMap::new(),
            pending_requests: HashMap::new(),
            event_tx: tx,
            started_at: chrono::Utc::now(),
            loaded_models: HashMap::new(),
            available_models: HashMap::new(),
            available_model_descriptors: HashMap::new(),
            queued_generation_by_session: HashMap::new(),
            model_ready_notify: Arc::new(Notify::new()),
            storage_root,
            gateway_tool_registry: Arc::new(ToolRegistry::new()),
            git_storage: None,
        }
    }

    /// Register a newly connected peer and its directed sender channel.
    pub fn add_peer(&mut self, info: PeerInfo, sender: mpsc::Sender<Frame>) {
        tracing::info!(
            "Peer connected: {} (role={:?}, client={}, device={:?}, scopes={}, connected_at={}, storage_root={})",
            info.id,
            info.role,
            info.client_id,
            info.device_id,
            info.scopes.len(),
            info.connected_at,
            self.storage_root.display()
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
        self.available_model_descriptors.remove(id);
    }

    /// Update the set of models available on disk for a peer.
    pub fn set_available_models(&mut self, peer_id: &str, models: Vec<String>) {
        self.available_models.insert(peer_id.to_string(), models);
    }

    /// Update the set of model descriptors available on disk for a peer.
    pub fn set_available_model_descriptors(&mut self, peer_id: &str, models: Vec<ModelDefinition>) {
        self.available_model_descriptors
            .insert(peer_id.to_string(), models);
        self.model_ready_notify.notify_waiters();
    }

    /// Update the loaded models for a node. Notifies queue drain waiters.
    pub fn set_loaded_models(&mut self, peer_id: &str, models: Vec<ModelDefinition>) {
        self.loaded_models.insert(peer_id.to_string(), models);
        self.model_ready_notify.notify_waiters();
    }

    /// Increment queued generation count for a session and return the new value.
    pub fn increment_generation_queue(&mut self, session_id: &str) -> usize {
        let entry = self
            .queued_generation_by_session
            .entry(session_id.to_string())
            .or_insert(0);
        *entry += 1;
        *entry
    }

    /// Decrement queued generation count for a session and return the remaining value.
    pub fn decrement_generation_queue(&mut self, session_id: &str) -> usize {
        let Some(count) = self.queued_generation_by_session.get_mut(session_id) else {
            return 0;
        };
        if *count <= 1 {
            self.queued_generation_by_session.remove(session_id);
            return 0;
        }

        *count -= 1;
        *count
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
                if peer.role != ConnectionRole::User
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

    /// Register tools provided by a node. Returns the number of tools registered.
    pub fn register_tools(&mut self, peer_id: &str, tools: Vec<ToolDefinition>) -> u32 {
        let count = tools.len() as u32;
        let now = chrono::Utc::now();
        for definition in tools {
            tracing::debug!("Registered tool '{}' from peer {peer_id}", definition.name,);
            self.node_tool_registry.insert(
                definition.name.clone(),
                RegisteredTool {
                    definition,
                    peer_id: peer_id.to_string(),
                    registered_at: now,
                },
            );
        }
        count
    }

    /// Remove all tools registered by a specific peer.
    pub fn deregister_tools_for_peer(&mut self, peer_id: &str) {
        let before = self.node_tool_registry.len();
        self.node_tool_registry.retain(|name, tool| {
            if tool.peer_id == peer_id {
                tracing::debug!(
                    "Deregistered tool '{name}' (peer {peer_id} disconnected, registered_at={})",
                    tool.registered_at
                );
                false
            } else {
                true
            }
        });
        let removed = before - self.node_tool_registry.len();
        if removed > 0 {
            tracing::info!("Deregistered {removed} tool(s) for peer {peer_id}");
        }
    }

    /// Build tool catalog entries from the registry (node tools + gateway-native tools).
    pub fn all_tool_entries(&self) -> Vec<ToolEntry> {
        let mut entries: Vec<ToolEntry> = self
            .node_tool_registry
            .values()
            .map(|rt| {
                ToolEntry::new(
                    rt.definition.clone(),
                    "node",
                    self.peer_senders.contains_key(&rt.peer_id),
                )
            })
            .collect();
        entries.extend(
            self.gateway_tool_registry
                .definitions()
                .into_iter()
                .map(|definition| ToolEntry::new(definition, "gateway", true)),
        );
        entries
    }

    /// Count currently connected user peers.
    pub fn connected_users(&self) -> u32 {
        self.peers
            .values()
            .filter(|p| p.role == ConnectionRole::User)
            .count() as u32
    }

    /// Count currently connected node peers.
    pub fn connected_nodes(&self) -> u32 {
        self.peers
            .values()
            .filter(|p| p.role == ConnectionRole::Node)
            .count() as u32
    }

    /// Return the deduplicated capability families exposed by connected nodes.
    pub fn all_capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = self
            .peers
            .values()
            .filter(|p| p.role == ConnectionRole::Node)
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

#[cfg(test)]
pub(crate) fn dummy_sender() -> mpsc::Sender<Frame> {
    let (tx, _rx) = mpsc::channel(1);
    tx
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use nexo_ws_schema::Scope;

    use super::*;

    fn make_user_peer(id: &str) -> PeerInfo {
        PeerInfo {
            id: id.into(),
            client_id: "cli".into(),
            role: ConnectionRole::User,
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
            role: ConnectionRole::Node,
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
                role: ConnectionRole::User,
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
                role: ConnectionRole::User,
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
    fn uptime_is_non_negative() {
        let state = GatewayState::new(std::path::PathBuf::from("/tmp"));
        assert!(state.uptime_secs() < 2);
    }
}
