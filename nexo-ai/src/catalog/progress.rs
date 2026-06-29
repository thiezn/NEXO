use std::sync::Arc;

use hf_hub::api::tokio::Progress;
use nexo_core::ModelId;

use super::model_file::ModelFileKind;

/// Receives high-level model and file download lifecycle events.
pub trait CatalogDownloadProgress: Send + Sync {
    /// Called before a model's files begin downloading.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model whose files are about to be downloaded.
    /// * `file_count` - The number of files planned for this model.
    /// * `total_bytes` - The total expected bytes across all planned files.
    fn model_started(&self, model_id: &ModelId, file_count: usize, total_bytes: u64);

    /// Creates a per-file progress sink.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model that owns the file being downloaded.
    /// * `kind` - The semantic kind of file being downloaded.
    /// * `repo` - The Hugging Face repository that hosts the file.
    /// * `remote_path` - The repository-relative remote file path.
    /// * `size_bytes` - The expected file size in bytes.
    fn file_started(
        &self,
        model_id: &ModelId,
        kind: ModelFileKind,
        repo: &str,
        remote_path: &str,
        size_bytes: u64,
    ) -> Arc<dyn FileDownloadProgress>;

    /// Called after all planned downloads for a model have completed successfully.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model whose downloads completed.
    fn model_finished(&self, model_id: &ModelId);
}

/// Receives byte-level progress updates for a single file download.
pub trait FileDownloadProgress: Send + Sync {
    /// Initializes the file progress display.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - The total file size in bytes reported by the downloader.
    /// * `label` - The user-facing label for the file progress item.
    fn init(&self, size_bytes: u64, label: &str);

    /// Advances the file progress display by a byte delta.
    ///
    /// # Arguments
    ///
    /// * `delta_bytes` - The number of bytes downloaded since the previous update.
    fn advance(&self, delta_bytes: u64);

    /// Marks the file progress display as finished.
    fn finish(&self);
}

/// No-op model progress sink used when callers do not provide UI feedback.
#[derive(Default)]
pub struct NoopDownloadProgress;

impl CatalogDownloadProgress for NoopDownloadProgress {
    fn model_started(&self, _model_id: &ModelId, _file_count: usize, _total_bytes: u64) {}

    fn file_started(
        &self,
        _model_id: &ModelId,
        _kind: ModelFileKind,
        _repo: &str,
        _remote_path: &str,
        _size_bytes: u64,
    ) -> Arc<dyn FileDownloadProgress> {
        Arc::new(NoopFileDownloadProgress)
    }

    fn model_finished(&self, _model_id: &ModelId) {}
}

#[derive(Default)]
struct NoopFileDownloadProgress;

impl FileDownloadProgress for NoopFileDownloadProgress {
    fn init(&self, _size_bytes: u64, _label: &str) {}

    fn advance(&self, _delta_bytes: u64) {}

    fn finish(&self) {}
}

/// Adapts a file progress sink to the `hf_hub` async progress interface.
#[derive(Clone)]
pub(crate) struct HfHubProgressAdapter {
    sink: Arc<dyn FileDownloadProgress>,
    label: String,
}

impl HfHubProgressAdapter {
    /// Creates a new `hf_hub` progress adapter.
    ///
    /// # Arguments
    ///
    /// * `sink` - The progress sink receiving byte updates for one file.
    /// * `label` - The user-facing label shown for the file.
    pub(crate) fn new(sink: Arc<dyn FileDownloadProgress>, label: String) -> Self {
        Self { sink, label }
    }
}

impl Progress for HfHubProgressAdapter {
    async fn init(&mut self, size: usize, filename: &str) {
        let label = if self.label.is_empty() {
            filename.to_string()
        } else {
            self.label.clone()
        };
        self.sink.init(size as u64, &label);
    }

    async fn update(&mut self, size: usize) {
        self.sink.advance(size as u64);
    }

    async fn finish(&mut self) {
        self.sink.finish();
    }
}
