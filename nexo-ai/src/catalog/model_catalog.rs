use super::ModelManifest;
use super::download::{CatalogDownloader, DownloadOptions};
use super::manifests::ALL_MANIFESTS;
use crate::Result;
use nexo_core::{ModelDefinition, ModelId};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::task::JoinSet;

/// The ModelCatalog manages the download and storage of model artifacts.
pub struct ModelCatalog {
    /// Any supported model manifest keyed by model ID.
    all_manifests: BTreeMap<ModelId, ModelManifest>,
}

impl ModelCatalog {
    /// Creates a new ModelCatalog with the static full list of ModelManifests.
    pub fn new() -> Self {
        Self {
            all_manifests: ALL_MANIFESTS
                .iter()
                .map(|m| {
                    let manifest: &ModelManifest = &**m;
                    (manifest.model_id().clone(), manifest.clone())
                })
                .collect(),
        }
    }

    /// Retrieves a model manifest by its ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to retrieve the manifest for.
    pub fn get_model_manifest(&self, model_id: &ModelId) -> Option<&ModelManifest> {
        self.all_manifests.get(model_id)
    }

    /// list all manifests, regardless of download status.
    pub fn list_all_manifests(&self) -> Vec<&ModelManifest> {
        self.all_manifests.values().collect()
    }

    /// Lists all downloaded model manifests.
    pub fn list_downloaded_manifests(&self) -> Vec<ModelManifest> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(manifest.clone()))
            .collect()
    }

    /// Retrieves a model definition by its ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to retrieve the definition for.
    pub fn get_model_definition(&self, model_id: &ModelId) -> Option<&ModelDefinition> {
        self.get_model_manifest(model_id)
            .map(|manifest| manifest.definition())
    }

    /// Lists all downloaded model definitions.
    pub fn list_available_models(&self) -> Vec<&ModelId> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(manifest.model_id()))
            .collect()
    }

    /// Lists all downloaded model definitions.
    pub fn list_available_model_definitions(&self) -> Vec<&ModelDefinition> {
        self.all_manifests
            .values()
            .filter_map(|manifest| manifest.is_downloaded().then_some(manifest.definition()))
            .collect()
    }

    /// Downloads a model manifest by its ID, returning an error if the model
    /// is not found or fails to download.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to download.
    pub async fn download_model(&self, model_id: &ModelId) -> Result {
        self.download_model_with_options(
            model_id,
            DownloadOptions::default(),
            Arc::new(crate::NoopDownloadProgress),
        )
        .await
    }

    /// Downloads a model manifest by its ID using explicit download options and progress reporting.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to download.
    /// * `options` - Download behavior options such as force mode and concurrency.
    /// * `progress` - The progress sink that receives model and file lifecycle events.
    pub async fn download_model_with_options(
        &self,
        model_id: &ModelId,
        options: DownloadOptions,
        progress: Arc<dyn crate::CatalogDownloadProgress>,
    ) -> Result {
        self.download_models_with_options(std::slice::from_ref(model_id), options, progress)
            .await
    }

    /// Downloads multiple model manifests using the default download options.
    ///
    /// # Arguments
    ///
    /// * `model_ids` - The model IDs that should be downloaded.
    pub async fn download_models(&self, model_ids: &[ModelId]) -> Result {
        self.download_models_with_options(
            model_ids,
            DownloadOptions::default(),
            Arc::new(crate::NoopDownloadProgress),
        )
        .await
    }

    /// Downloads multiple model manifests using explicit download options and progress reporting.
    ///
    /// # Arguments
    ///
    /// * `model_ids` - The model IDs that should be downloaded.
    /// * `options` - Download behavior options such as force mode and concurrency.
    /// * `progress` - The progress sink that receives model and file lifecycle events.
    pub async fn download_models_with_options(
        &self,
        model_ids: &[ModelId],
        options: DownloadOptions,
        progress: Arc<dyn crate::CatalogDownloadProgress>,
    ) -> Result {
        let downloader = CatalogDownloader::new(options)?;
        let max_concurrent_files = downloader.options().max_concurrent_files.max(1);
        let mut remaining_files_by_model = BTreeMap::new();
        let mut pending_files = Vec::new();

        for requested_model_id in model_ids {
            let manifest = self.get_model_manifest(requested_model_id).ok_or_else(|| {
                crate::Error::UnknownModel {
                    model_id: requested_model_id.clone(),
                }
            })?;

            let model_id = manifest.model_id().clone();
            let model_dir = manifest.model_dir()?;
            let files = manifest
                .files()
                .iter()
                .filter(|file| downloader.options().force || !file.is_downloaded(&model_dir))
                .cloned()
                .collect::<Vec<_>>();
            let total_bytes = files.iter().map(|file| file.size_bytes()).sum();

            progress.model_started(&model_id, files.len(), total_bytes);

            if files.is_empty() {
                progress.model_finished(&model_id);
                continue;
            }

            remaining_files_by_model.insert(model_id.clone(), files.len());
            pending_files.extend(
                files
                    .into_iter()
                    .map(|file| (model_id.clone(), model_dir.clone(), file)),
            );
        }

        if pending_files.is_empty() {
            return Ok(());
        }

        let mut pending_files = pending_files.into_iter();
        let mut in_flight = JoinSet::new();

        for _ in 0..max_concurrent_files {
            if !spawn_next_file_download(
                &mut in_flight,
                &mut pending_files,
                &downloader,
                progress.clone(),
            ) {
                break;
            }
        }

        while let Some(result) = in_flight.join_next().await {
            let finished_model_id = result.map_err(|error| crate::Error::Runtime {
                message: format!("model download task failed: {error}"),
            })??;

            if let Some(remaining_files) = remaining_files_by_model.get_mut(&finished_model_id) {
                *remaining_files -= 1;
                if *remaining_files == 0 {
                    remaining_files_by_model.remove(&finished_model_id);
                    progress.model_finished(&finished_model_id);
                }
            }

            let _ = spawn_next_file_download(
                &mut in_flight,
                &mut pending_files,
                &downloader,
                progress.clone(),
            );
        }

        Ok(())
    }
}

fn spawn_next_file_download(
    in_flight: &mut JoinSet<Result<ModelId>>,
    pending_files: &mut impl Iterator<Item = (ModelId, std::path::PathBuf, super::ModelFile)>,
    downloader: &CatalogDownloader,
    progress: Arc<dyn crate::CatalogDownloadProgress>,
) -> bool {
    let Some((model_id, model_dir, file)) = pending_files.next() else {
        return false;
    };

    let downloader = downloader.clone();
    in_flight.spawn(async move {
        let file_progress = progress.file_started(
            &model_id,
            file.kind(),
            file.hf_repo(),
            file.remote_path(),
            file.size_bytes(),
        );
        file.download(&downloader, &model_dir, file_progress)
            .await?;
        Ok(model_id)
    });

    true
}
