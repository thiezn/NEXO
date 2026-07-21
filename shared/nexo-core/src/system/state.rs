use crate::{Error, Node, PeerId, Result, User};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The state of the Nexo System.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct NexoState {
    /// The list of currently active nodes in the Nexo system.
    nodes: HashMap<PeerId, Node>,

    /// The list of currently active users in the Nexo system.
    users: HashMap<PeerId, User>,
}

impl NexoState {
    /// Creates a new NexoState instance.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            users: HashMap::new(),
        }
    }

    /// Adds a new user to the Nexo system state.
    pub fn add_user(&mut self, user: User) -> Result {
        if let Some(_) = self.users.get(&user.id()) {
            return Err(Error::PeerAlreadyConnected {
                message: format!(
                    "User with ID {} already exists with different properties",
                    user.id()
                ),
            });
        }

        self.users.insert(user.id(), user);
        Ok(())
    }

    /// Adds a new node to the Nexo system state.
    pub fn add_node(&mut self, node: Node) -> Result {
        if let Some(_) = self.nodes.get(&node.id()) {
            return Err(Error::PeerAlreadyConnected {
                message: format!(
                    "Node with ID {} already exists with different properties",
                    node.id()
                ),
            });
        }

        self.nodes.insert(node.id(), node);
        Ok(())
    }

    /// Removes a user from the Nexo system state.
    pub fn remove_user(&mut self, user_id: &PeerId) {
        self.users.remove(user_id);
    }

    /// Removes a node from the Nexo system state.
    pub fn remove_node(&mut self, node_id: &PeerId) {
        self.nodes.remove(node_id);
    }

    /// Returns the number of active nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the currently active nodes keyed by peer identifier.
    pub fn nodes(&self) -> &HashMap<PeerId, Node> {
        &self.nodes
    }

    /// Returns the number of active users.
    pub fn user_count(&self) -> usize {
        self.users.len()
    }
}
