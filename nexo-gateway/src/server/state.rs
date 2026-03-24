use nexo_ws_schema::{Frame, Role, Scope};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

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

/// Shared mutable state for the gateway.
pub struct GatewayState {
    pub peers: HashMap<PeerId, PeerInfo>,
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
            event_tx: tx,
            started_at: chrono::Utc::now(),
        }
    }

    pub fn add_peer(&mut self, info: PeerInfo) {
        tracing::info!(
            "Peer connected: {} (role={:?}, client={})",
            info.id,
            info.role,
            info.client_id
        );
        self.peers.insert(info.id.clone(), info);
    }

    pub fn remove_peer(&mut self, id: &str) {
        if let Some(peer) = self.peers.remove(id) {
            tracing::info!("Peer disconnected: {} (client={})", peer.id, peer.client_id);
        }
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
}

pub type SharedState = Arc<RwLock<GatewayState>>;

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
        state.add_peer(make_user_peer("peer-1"));
        assert_eq!(state.peers.len(), 1);
        state.remove_peer("peer-1");
        assert!(state.peers.is_empty());
    }

    #[test]
    fn connected_users_and_nodes_count() {
        let mut state = GatewayState::new();
        state.add_peer(make_user_peer("u1"));
        state.add_peer(make_user_peer("u2"));
        state.add_peer(make_node_peer("n1", vec!["epub".into()]));
        assert_eq!(state.connected_users(), 2);
        assert_eq!(state.connected_nodes(), 1);
    }

    #[test]
    fn all_capabilities_aggregation() {
        let mut state = GatewayState::new();
        state.add_peer(make_node_peer("n1", vec!["epub".into(), "game".into()]));
        state.add_peer(make_node_peer("n2", vec!["game".into(), "tts".into()]));
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
}
