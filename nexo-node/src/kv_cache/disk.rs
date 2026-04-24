use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use nexo_ai::api::model_traits::KvCacheable;

use super::types::CacheMetadata;

/// Persist a model's current KV cache to disk using nexo-ai's cache format.
pub fn save_to_disk(dir: &Path, model_name: &str, model: &dyn KvCacheable) -> Result<bool> {
    nexo_ai::api::kv_cache::save_model_cache_to_disk(dir, model_name, model)
}

/// Restore a model session's KV cache from disk.
pub fn load_from_disk(
    dir: &Path,
    session_id: &str,
    model: &mut dyn KvCacheable,
) -> Result<Option<CacheMetadata>> {
    nexo_ai::api::kv_cache::load_model_cache_from_disk(dir, session_id, model)
}

/// Delete a session's cache files from disk.
pub fn delete_cache(dir: &Path, session_id: &str) -> Result<()> {
    nexo_ai::api::kv_cache::delete_model_cache_from_disk(dir, session_id)
}

/// Remove old caches that exceed max_entries or max_age.
/// Returns the number of caches deleted.
pub fn expire_old_caches(dir: &Path, max_entries: usize, max_age: Duration) -> Result<usize> {
    nexo_ai::api::kv_cache::expire_model_caches_on_disk(dir, max_entries, max_age)
}
