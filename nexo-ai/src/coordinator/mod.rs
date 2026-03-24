pub mod load;
pub mod registry;
pub mod unload;

use crate::config::AiConfig;
use crate::shared::model_traits::ModelInfo;
use crate::shared::types::ModelCategory;
use std::collections::HashMap;

pub struct ModelSlot {
    model: Box<dyn ModelInfo>,
    categories: Vec<ModelCategory>,
}

impl ModelSlot {
    pub fn new(model: Box<dyn ModelInfo>, categories: Vec<ModelCategory>) -> Self {
        Self { model, categories }
    }
    pub fn name(&self) -> &str {
        self.model.name()
    }
    pub fn categories(&self) -> &[ModelCategory] {
        &self.categories
    }
    pub fn is_loaded(&self) -> bool {
        self.model.is_loaded()
    }
    pub fn memory_estimate_bytes(&self) -> u64 {
        self.model.memory_estimate_bytes()
    }
    pub fn model_mut(&mut self) -> &mut dyn ModelInfo {
        &mut *self.model
    }
}

pub struct Coordinator {
    config: AiConfig,
    slots: HashMap<String, ModelSlot>,
    active_defaults: HashMap<ModelCategory, String>,
}

impl Coordinator {
    pub fn new(config: AiConfig) -> Self {
        let active_defaults = config
            .defaults
            .iter()
            .filter_map(|(cat_str, model_name)| {
                ModelCategory::all()
                    .iter()
                    .find(|c| c.as_str() == cat_str)
                    .map(|c| (*c, model_name.clone()))
            })
            .collect();
        Self {
            config,
            slots: HashMap::new(),
            active_defaults,
        }
    }

    pub fn config(&self) -> &AiConfig {
        &self.config
    }
    pub fn config_mut(&mut self) -> &mut AiConfig {
        &mut self.config
    }

    pub fn loaded_models(&self) -> Vec<(&str, &[ModelCategory])> {
        self.slots
            .values()
            .filter(|s| s.is_loaded())
            .map(|s| (s.name(), s.categories()))
            .collect()
    }

    pub fn default_for(&self, category: ModelCategory) -> Option<&str> {
        self.active_defaults.get(&category).map(String::as_str)
    }

    pub fn set_default(&mut self, category: ModelCategory, model_name: String) {
        self.active_defaults.insert(category, model_name);
    }

    pub fn total_memory_used(&self) -> u64 {
        self.slots
            .values()
            .filter(|s| s.is_loaded())
            .map(|s| s.memory_estimate_bytes())
            .sum()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::models::stub::StubModel;

    fn config_with_defaults() -> AiConfig {
        let mut config = AiConfig::default();
        config
            .defaults
            .insert("chat".to_string(), "test-chat".to_string());
        config
            .defaults
            .insert("image".to_string(), "test-image".to_string());
        config
    }

    #[test]
    fn new_coordinator_parses_defaults() {
        let coord = Coordinator::new(config_with_defaults());
        assert_eq!(coord.active_defaults.len(), 2);
        assert_eq!(
            coord.active_defaults.get(&ModelCategory::Chat).unwrap(),
            "test-chat"
        );
        assert_eq!(
            coord.active_defaults.get(&ModelCategory::Image).unwrap(),
            "test-image"
        );
    }

    #[test]
    fn default_for_returns_correct_model() {
        let coord = Coordinator::new(config_with_defaults());
        assert_eq!(coord.default_for(ModelCategory::Chat), Some("test-chat"));
        assert_eq!(coord.default_for(ModelCategory::Image), Some("test-image"));
        assert_eq!(coord.default_for(ModelCategory::Talk), None);
    }

    #[test]
    fn set_default_updates_active_default() {
        let mut coord = Coordinator::new(AiConfig::default());
        assert!(coord.default_for(ModelCategory::Chat).is_none());

        coord.set_default(ModelCategory::Chat, "new-model".to_string());
        assert_eq!(coord.default_for(ModelCategory::Chat), Some("new-model"));

        coord.set_default(ModelCategory::Chat, "newer-model".to_string());
        assert_eq!(coord.default_for(ModelCategory::Chat), Some("newer-model"));
    }

    #[test]
    fn loaded_models_empty_when_nothing_loaded() {
        let coord = Coordinator::new(AiConfig::default());
        assert!(coord.loaded_models().is_empty());
    }

    #[test]
    fn total_memory_used_zero_when_nothing_loaded() {
        let coord = Coordinator::new(AiConfig::default());
        assert_eq!(coord.total_memory_used(), 0);
    }

    #[test]
    fn unload_all_clears_slots() {
        let mut coord = Coordinator::new(AiConfig::default());
        let mut stub = StubModel::new("m1", 1_000_000);
        stub.load().unwrap();
        coord.slots.insert(
            "m1".to_string(),
            ModelSlot::new(Box::new(stub), vec![ModelCategory::Chat]),
        );
        assert_eq!(coord.slots.len(), 1);

        coord.unload_all();
        assert!(coord.slots.is_empty());
    }

    #[test]
    fn free_memory_with_loaded_stubs() {
        let mut coord = Coordinator::new(AiConfig::default());

        let mut s1 = StubModel::new("small", 1_000_000);
        s1.load().unwrap();
        coord.slots.insert(
            "small".to_string(),
            ModelSlot::new(Box::new(s1), vec![ModelCategory::Chat]),
        );

        let mut s2 = StubModel::new("large", 5_000_000);
        s2.load().unwrap();
        coord.slots.insert(
            "large".to_string(),
            ModelSlot::new(Box::new(s2), vec![ModelCategory::Image]),
        );

        assert_eq!(coord.total_memory_used(), 6_000_000);

        // Free at least 4MB -- should unload "large" first (sorted by largest)
        let freed = coord.free_memory(4_000_000).unwrap();
        assert!(freed >= 4_000_000);
    }
}
