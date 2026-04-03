use serde::{Deserialize, Serialize};

/// Metadata sidecar stored alongside safetensors KV cache files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub session_id: String,
    pub model_name: String,
    pub processed_tokens: Vec<u32>,
    pub layer_count: usize,
    pub created_at: String,
    pub last_accessed: String,
}
