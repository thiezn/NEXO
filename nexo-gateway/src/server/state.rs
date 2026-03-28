use nexo_ws_schema::{Frame, Role, Scope, ToolEntry, ToolSpecEntry};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc, oneshot};

pub type PeerId = String;

/// Information about a connected peer (user or node).
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: PeerId,
    pub client_id: String,
    pub role: Role,
    pub scopes: Vec<Scope>,
    pub capabilities: Vec<String>,
    pub commands: Vec<String>,
    pub device_id: Option<String>,
    pub connected_at: chrono::DateTime<chrono::Utc>,
}

/// A tool registered by a node.
#[derive(Debug, Clone)]
pub struct RegisteredTool {
    pub spec: ToolSpecEntry,
    pub peer_id: PeerId,
    pub registered_at: chrono::DateTime<chrono::Utc>,
}

/// Shared mutable state for the gateway.
pub struct GatewayState {
    pub peers: HashMap<PeerId, PeerInfo>,
    pub peer_senders: HashMap<PeerId, mpsc::Sender<Frame>>,
    pub tool_registry: HashMap<String, RegisteredTool>,
    pub pending_requests: HashMap<String, oneshot::Sender<Frame>>,
    pub event_tx: broadcast::Sender<Frame>,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl Default for GatewayState {
    fn default() -> Self {
        Self::new()
    }
}

impl GatewayState {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            peers: HashMap::new(),
            peer_senders: HashMap::new(),
            tool_registry: HashMap::new(),
            pending_requests: HashMap::new(),
            event_tx: tx,
            started_at: chrono::Utc::now(),
        }
    }

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

    pub fn remove_peer(&mut self, id: &str) {
        if let Some(peer) = self.peers.remove(id) {
            tracing::info!("Peer disconnected: {} (client={})", peer.id, peer.client_id);
        }
        self.peer_senders.remove(id);
        self.deregister_tools_for_peer(id);
    }

    /// Register tools provided by a node. Returns the number of tools registered.
    pub fn register_tools(&mut self, peer_id: &str, tools: Vec<ToolSpecEntry>) -> u32 {
        let count = tools.len() as u32;
        let now = chrono::Utc::now();
        for spec in tools {
            tracing::debug!(
                "Registered tool '{}' from peer {peer_id}",
                spec.name,
            );
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
            tracing::info!(
                "Deregistered {removed} tool(s) for peer {peer_id}"
            );
        }
    }

    /// Look up a registered tool by name.
    pub fn find_tool(&self, name: &str) -> Option<&RegisteredTool> {
        self.tool_registry.get(name)
    }

    /// Build tool catalog entries from the registry.
    pub fn all_tool_entries(&self) -> Vec<ToolEntry> {
        self.tool_registry
            .values()
            .map(|rt| ToolEntry {
                name: rt.spec.name.clone(),
                description: rt.spec.description.clone(),
                source: "node".into(),
                available: self.peer_senders.contains_key(&rt.peer_id),
                parameters: Some(rt.spec.parameters.clone()),
            })
            .collect()
    }

    pub fn connected_users(&self) -> u32 {
        self.peers.values().filter(|p| p.role == Role::User).count() as u32
    }

    pub fn connected_nodes(&self) -> u32 {
        self.peers.values().filter(|p| p.role == Role::Node).count() as u32
    }

    pub fn all_capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = self
            .peers
            .values()
            .filter(|p| p.role == Role::Node)
            .flat_map(|p| p.capabilities.clone())
            .collect();
        caps.sort();
        caps.dedup();
        caps
    }

    pub fn uptime_secs(&self) -> u64 {
        (chrono::Utc::now() - self.started_at).num_seconds().max(0) as u64
    }

    /// Get the client_id (used as user_id) for a peer.
    pub fn user_id_for_peer(&self, peer_id: &str) -> Option<String> {
        self.peers.get(peer_id).map(|p| p.client_id.clone())
    }
}

pub type SharedState = Arc<RwLock<GatewayState>>;

#[cfg(test)]
pub(crate) fn dummy_sender() -> mpsc::Sender<Frame> {
    let (tx, _rx) = mpsc::channel(1);
    tx
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

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
        let mut state = GatewayState::new();
        state.add_peer(make_user_peer("peer-1"), dummy_sender());
        assert_eq!(state.peers.len(), 1);
        state.remove_peer("peer-1");
        assert!(state.peers.is_empty());
    }

    #[test]
    fn connected_users_and_nodes_count() {
        let mut state = GatewayState::new();
        state.add_peer(make_user_peer("u1"), dummy_sender());
        state.add_peer(make_user_peer("u2"), dummy_sender());
        state.add_peer(make_node_peer("n1", vec!["epub".into()]), dummy_sender());
        assert_eq!(state.connected_users(), 2);
        assert_eq!(state.connected_nodes(), 1);
    }

    #[test]
    fn all_capabilities_aggregation() {
        let mut state = GatewayState::new();
        state.add_peer(make_node_peer("n1", vec!["epub".into(), "game".into()]), dummy_sender());
        state.add_peer(make_node_peer("n2", vec!["game".into(), "tts".into()]), dummy_sender());
        let caps = state.all_capabilities();
        assert_eq!(caps, vec!["epub", "game", "tts"]);
    }

    #[test]
    fn all_capabilities_empty_with_no_nodes() {
        let state = GatewayState::new();
        assert!(state.all_capabilities().is_empty());
    }

    #[test]
    fn uptime_is_non_negative() {
        let state = GatewayState::new();
        assert!(state.uptime_secs() < 2);
    }

    #[test]
    fn register_and_deregister_tools() {
        let mut state = GatewayState::new();
        state.add_peer(make_node_peer("n1", vec!["echo".into()]), dummy_sender());

        let tools = vec![
            ToolSpecEntry {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolSpecEntry {
                name: "echo.ping".into(),
                description: "Ping".into(),
                parameters: serde_json::json!({"type": "object"}),
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
        let mut state = GatewayState::new();
        state.add_peer(make_node_peer("n1", vec!["echo".into()]), dummy_sender());
        state.register_tools(
            "n1",
            vec![ToolSpecEntry {
                name: "echo.run".into(),
                description: "Echo input".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        );

        let entries = state.all_tool_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "echo.run");
        assert_eq!(entries[0].source, "node");
        assert!(entries[0].available);
        assert!(entries[0].parameters.is_some());
    }

    #[test]
    fn tool_from_disconnected_peer_shows_unavailable() {
        let mut state = GatewayState::new();
        // Manually insert a tool without a matching peer sender
        state.tool_registry.insert(
            "orphan.tool".into(),
            RegisteredTool {
                spec: ToolSpecEntry {
                    name: "orphan.tool".into(),
                    description: "Orphan".into(),
                    parameters: serde_json::json!({}),
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
