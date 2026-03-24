use super::manifest::known_manifests;
use crate::download::paths::model_storage_dir;
use crate::shared::types::ModelCategory;

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub family: String,
    pub categories: Vec<ModelCategory>,
    pub size_gb: f32,
    pub description: String,
    pub is_downloaded: bool,
    pub is_loaded: bool,
}

/// Build a list of all known models, checking download status and
/// using the provided closure to determine if each model is currently loaded.
pub fn list_models(is_loaded: impl Fn(&str) -> bool) -> Vec<ModelEntry> {
    known_manifests()
        .iter()
        .map(|m| {
            let name = &m.manifest.name;
            ModelEntry {
                name: name.clone(),
                family: m.manifest.family.clone(),
                categories: m.categories.clone(),
                size_gb: m.manifest.size_gb,
                description: m.manifest.description.clone(),
                is_downloaded: model_storage_dir(name).exists(),
                is_loaded: is_loaded(name),
            }
        })
        .collect()
}
