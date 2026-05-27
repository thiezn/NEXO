use std::collections::BTreeMap;

use serde_json::Value;

/// Arbitrary JSON metadata attached to requests, messages, models, or runs.
pub type MetadataMap = BTreeMap<String, Value>;
