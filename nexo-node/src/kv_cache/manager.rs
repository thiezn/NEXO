use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::{DType, Device};
use nexo_ai::shared::model_traits::KvCacheable;

use super::disk;

const DEFAULT_MAX_DISK_ENTRIES: usize = 8;
const DEFAULT_MAX_CACHE_AGE: Duration = Duration::from_secs(3600); // 1 hour

/// Manages KV cache persistence across session switches.
///
/// Sits at the nexo-node level and coordinates between the in-memory KV cache
/// (managed by the pipeline inside nexo-ai) and disk storage.
pub struct SessionCacheManager {
    cache_dir: PathBuf,
    max_disk_entries: usize,
    max_cache_age: Duration,
    last_expire: Option<Instant>,
}

impl SessionCacheManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            max_disk_entries: DEFAULT_MAX_DISK_ENTRIES,
            max_cache_age: DEFAULT_MAX_CACHE_AGE,
            last_expire: None,
        }
    }

    /// Save the current in-memory session cache to disk.
    /// Called before switching to a different session.
    pub fn save_current_session(
        &self,
        model_name: &str,
        model: &dyn KvCacheable,
    ) -> Result<()> {
        let Some(session_id) = model.current_session_id() else {
            tracing::debug!("KV cache: no current session to save");
            return Ok(());
        };

        if model.kv_cache_seq_len() == 0 {
            tracing::debug!("KV cache: empty cache for session {session_id}, skipping save");
            return Ok(());
        }

        let dir = self.cache_dir.join(model_name);
        let snapshots = model.save_kv_cache()?;

        disk::save_to_disk(
            &dir,
            session_id,
            model_name,
            &snapshots,
            model.processed_tokens(),
        )?;

        Ok(())
    }

    /// Try to load a session's cache from disk and restore it into the model.
    /// Returns true if cache was restored, false if not found on disk.
    pub fn load_session(
        &self,
        session_id: &str,
        model_name: &str,
        model: &mut dyn KvCacheable,
        device: &Device,
        dtype: DType,
    ) -> Result<bool> {
        let dir = self.cache_dir.join(model_name);

        let Some((snapshots, metadata)) = disk::load_from_disk(&dir, session_id, device, dtype)?
        else {
            tracing::debug!("KV cache: no disk cache found for session {session_id}");
            return Ok(false);
        };

        // Restore KV cache into the model
        model.restore_kv_cache(&snapshots)?;
        model.set_session_state(
            Some(metadata.session_id),
            metadata.processed_tokens,
        );

        tracing::debug!(
            "KV cache: restored session {session_id} from disk ({} layers, {} tokens)",
            snapshots.len(),
            model.processed_tokens().len(),
        );

        Ok(true)
    }

    /// Called before model unload — save current session so it can be
    /// restored later if the model is reloaded.
    pub fn on_model_unload(
        &self,
        model_name: &str,
        model: &dyn KvCacheable,
    ) -> Result<()> {
        self.save_current_session(model_name, model)
    }

    /// Run cache expiry. Called periodically (e.g., after each inference).
    pub fn maybe_expire(&mut self) -> Result<()> {
        // Only run expiry every 5 minutes
        if let Some(last) = self.last_expire {
            if last.elapsed() < Duration::from_secs(300) {
                return Ok(());
            }
        }

        self.last_expire = Some(Instant::now());

        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    disk::expire_old_caches(
                        &entry.path(),
                        self.max_disk_entries,
                        self.max_cache_age,
                    )?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_manager() {
        let mgr = SessionCacheManager::new(PathBuf::from("/tmp/test_kv_cache"));
        assert_eq!(mgr.max_disk_entries, DEFAULT_MAX_DISK_ENTRIES);
        assert_eq!(mgr.max_cache_age, DEFAULT_MAX_CACHE_AGE);
    }
}
