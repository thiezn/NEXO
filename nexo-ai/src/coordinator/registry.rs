use crate::download::{default_models_dir, known_manifests};
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

impl super::Coordinator {
    pub fn list_models(&self) -> Vec<ModelEntry> {
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
                    is_downloaded: self.check_downloaded(name),
                    is_loaded: self.slots.get(name.as_str()).is_some_and(|s| s.is_loaded()),
                }
            })
            .collect()
    }

    fn check_downloaded(&self, model_name: &str) -> bool {
        // Check if model directory exists under models dir
        let models_dir = default_models_dir();
        let model_dir = models_dir.join(model_name);
        model_dir.exists()
    }
}
