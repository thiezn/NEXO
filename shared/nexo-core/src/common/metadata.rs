use std::collections::HashMap;

use serde_json::Value;

/// Arbitrary JSON metadata attached to requests, messages, models, or runs.
pub type MetadataMap = HashMap<String, Value>;
