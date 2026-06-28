use crate::SessionId;
use serde::{Deserialize, Serialize};

/// A list of Inference sessions in the Nexo system.
pub type Sessions = Vec<Session>;

/// A single session entry in a session list response.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct Session {
    /// Field value.
    pub session_id: SessionId,
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
