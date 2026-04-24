use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use nexo_ai::api::model_traits::KvCacheable;

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
    pub fn save_current_session(&self, model_name: &str, model: &dyn KvCacheable) -> Result<()> {
        let dir = self.cache_dir.join(model_name);
        let _ = disk::save_to_disk(&dir, model_name, model)?;

        Ok(())
    }

    /// Try to load a session's cache from disk and restore it into the model.
    /// Returns true if cache was restored, false if not found on disk.
    pub fn load_session(
        &self,
        session_id: &str,
        model_name: &str,
        model: &mut dyn KvCacheable,
    ) -> Result<bool> {
        let dir = self.cache_dir.join(model_name);

        let Some(metadata) = disk::load_from_disk(&dir, session_id, model)? else {
            tracing::debug!("KV cache: no disk cache found for session {session_id}");
            return Ok(false);
        };

        tracing::debug!(
            "KV cache: restored session {session_id} from disk ({} layers, {} tokens)",
            metadata.layer_count,
            model.processed_tokens().len(),
        );

        Ok(true)
    }

    /// Switch the in-memory model cache to the target session.
    ///
    /// This handles persisting the current session, restoring a target session
    /// if available, or clearing state when starting fresh / going stateless.
    pub fn switch_session(
        &self,
        model_name: &str,
        model: &mut dyn KvCacheable,
        target_session_id: Option<&str>,
    ) -> Result<()> {
        let current_session_id = model.current_session_id().map(str::to_owned);
        if current_session_id.as_deref() == target_session_id {
            return Ok(());
        }

        if current_session_id.is_some() {
            self.save_current_session(model_name, model)?;
        }

        match target_session_id {
            Some(session_id) => {
                if !self.load_session(session_id, model_name, model)? {
                    model.clear_kv_cache();
                    model.set_session_state(Some(session_id.to_string()), Vec::new());
                    tracing::debug!(
                        "KV cache: no disk cache for session {session_id}, starting fresh"
                    );
                }
            }
            None => {
                model.clear_kv_cache();
                model.set_session_state(None, Vec::new());
                tracing::debug!("KV cache: cleared in-memory state for sessionless request");
            }
        }

        Ok(())
    }

    /// Called before model unload — save current session so it can be
    /// restored later if the model is reloaded.
    pub fn on_model_unload(&self, model_name: &str, model: &dyn KvCacheable) -> Result<()> {
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
    use candle_core::{DType, Device, Tensor};
    use nexo_ai::api::types::LayerKvSnapshot;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "nexo-node-kv-cache-{label}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    struct FakeKvModel {
        session_id: Option<String>,
        processed_tokens: Vec<u32>,
        seq_len: usize,
        clear_count: usize,
        device: Device,
    }

    impl FakeKvModel {
        fn new(session_id: Option<&str>, processed_tokens: Vec<u32>, seq_len: usize) -> Self {
            Self {
                session_id: session_id.map(str::to_owned),
                processed_tokens,
                seq_len,
                clear_count: 0,
                device: Device::Cpu,
            }
        }
    }

    impl KvCacheable for FakeKvModel {
        fn kv_cache_seq_len(&self) -> usize {
            self.seq_len
        }

        fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>> {
            let tensor = Tensor::zeros((1, 1, self.seq_len.max(1), 1), DType::F32, &self.device)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            Ok(vec![LayerKvSnapshot {
                layer_idx: 0,
                is_sliding: false,
                k_data: Some(tensor.clone()),
                v_data: Some(tensor),
                offset: 0,
                current_seq_len: self.seq_len,
            }])
        }

        fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()> {
            self.seq_len = snapshots
                .first()
                .map(|snap| snap.current_seq_len)
                .unwrap_or(0);
            Ok(())
        }

        fn clear_kv_cache(&mut self) {
            self.clear_count += 1;
            self.seq_len = 0;
        }

        fn processed_tokens(&self) -> &[u32] {
            &self.processed_tokens
        }

        fn current_session_id(&self) -> Option<&str> {
            self.session_id.as_deref()
        }

        fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>) {
            self.session_id = session_id;
            self.processed_tokens = tokens;
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn dtype(&self) -> DType {
            DType::F32
        }
    }

    #[test]
    fn new_creates_manager() {
        let mgr = SessionCacheManager::new(PathBuf::from("/tmp/test_kv_cache"));
        assert_eq!(mgr.max_disk_entries, DEFAULT_MAX_DISK_ENTRIES);
        assert_eq!(mgr.max_cache_age, DEFAULT_MAX_CACHE_AGE);
    }

    #[test]
    fn switch_session_to_none_clears_state() {
        let tmp = TempDir::new("clear-none");
        let mgr = SessionCacheManager::new(tmp.path.clone());
        let mut model = FakeKvModel::new(Some("session-a"), vec![1, 2, 3], 3);

        mgr.switch_session("test-model", &mut model, None).unwrap();

        assert_eq!(model.current_session_id(), None);
        assert!(model.processed_tokens().is_empty());
        assert_eq!(model.kv_cache_seq_len(), 0);
        assert_eq!(model.clear_count, 1);
    }

    #[test]
    fn switch_session_to_missing_entry_starts_fresh() {
        let tmp = TempDir::new("missing-session");
        let mgr = SessionCacheManager::new(tmp.path.clone());
        let mut model = FakeKvModel::new(Some("session-a"), vec![1, 2, 3], 3);

        mgr.switch_session("test-model", &mut model, Some("session-b"))
            .unwrap();

        assert_eq!(model.current_session_id(), Some("session-b"));
        assert!(model.processed_tokens().is_empty());
        assert_eq!(model.kv_cache_seq_len(), 0);
        assert_eq!(model.clear_count, 1);
    }

    #[test]
    fn switch_session_restores_saved_cache() {
        let tmp = TempDir::new("restore-session");
        let mgr = SessionCacheManager::new(tmp.path.clone());
        let mut model = FakeKvModel::new(Some("session-a"), vec![1, 2, 3], 3);

        mgr.save_current_session("test-model", &model).unwrap();
        model.set_session_state(Some("session-b".to_string()), vec![9]);
        model.seq_len = 1;

        mgr.switch_session("test-model", &mut model, Some("session-a"))
            .unwrap();

        assert_eq!(model.current_session_id(), Some("session-a"));
        assert_eq!(model.processed_tokens(), &[1, 2, 3]);
        assert_eq!(model.kv_cache_seq_len(), 3);
        assert_eq!(model.clear_count, 1);
    }
}
