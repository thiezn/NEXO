pub mod load;
pub mod unload;

use crate::config::AiConfig;
use crate::shared::model_traits::ModelInfo;
use crate::shared::types::ModelCategory;
use crate::statistics::StatsCollector;
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
    active_models: HashMap<ModelCategory, String>,
    stats: StatsCollector,
}

impl Coordinator {
    pub fn new(config: AiConfig) -> Self {
        let active_models = config
            .active_models
            .iter()
            .filter_map(|(cat_str, model_name)| {
                cat_str
                    .parse::<ModelCategory>()
                    .ok()
                    .map(|c| (c, model_name.clone()))
            })
            .collect();

        Self {
            config,
            slots: HashMap::new(),
            active_models,
            stats: StatsCollector::new(),
        }
    }

    pub fn config(&self) -> &AiConfig {
        &self.config
    }
    pub fn config_mut(&mut self) -> &mut AiConfig {
        &mut self.config
    }

    pub fn stats(&self) -> &StatsCollector {
        &self.stats
    }
    pub fn stats_mut(&mut self) -> &mut StatsCollector {
        &mut self.stats
    }

    pub fn loaded_models(&self) -> Vec<(&str, &[ModelCategory])> {
        self.slots
            .values()
            .filter(|s| s.is_loaded())
            .map(|s| (s.name(), s.categories()))
            .collect()
    }

    pub fn loaded_model_count(&self) -> usize {
        self.slots.values().filter(|s| s.is_loaded()).count()
    }

    pub fn active_model_for(&self, category: ModelCategory) -> Option<&str> {
        self.active_models.get(&category).map(String::as_str)
    }

    pub fn set_active_model(&mut self, category: ModelCategory, model_name: String) {
        self.active_models.insert(category, model_name.clone());
        self.config.set_active_model(category, model_name);
        if let Err(e) = self.config.save() {
            tracing::warn!("failed to persist config: {e}");
        }
    }

    pub fn remove_active_model(&mut self, category: ModelCategory) {
        self.active_models.remove(&category);
        self.config.remove_active_model(category);
        if let Err(e) = self.config.save() {
            tracing::warn!("failed to persist config: {e}");
        }
    }

    pub fn clear_active_models(&mut self) {
        self.active_models.clear();
        self.config.clear_active_models();
        if let Err(e) = self.config.save() {
            tracing::warn!("failed to persist config: {e}");
        }
    }

    pub fn is_model_loaded(&self, name: &str) -> bool {
        self.slots.get(name).is_some_and(|s| s.is_loaded())
    }

    pub fn list_models(&self) -> Vec<crate::registry::ModelEntry> {
        crate::registry::list_models(|name| self.is_model_loaded(name))
    }

    pub fn model_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut dyn crate::shared::model_traits::ModelInfo> {
        self.slots.get_mut(name).map(|s| s.model_mut())
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

    fn config_with_active_models() -> AiConfig {
        let mut config = AiConfig::default();
        config
            .active_models
            .insert("chat".to_string(), "test-chat".to_string());
        config
            .active_models
            .insert("image".to_string(), "test-image".to_string());
        config
    }

    #[test]
    fn new_coordinator_parses_active_models() {
        let coord = Coordinator::new(config_with_active_models());
        assert_eq!(coord.active_models.len(), 2);
        assert_eq!(
            coord.active_models.get(&ModelCategory::Chat).unwrap(),
            "test-chat"
        );
        assert_eq!(
            coord.active_models.get(&ModelCategory::Image).unwrap(),
            "test-image"
        );
    }

    #[test]
    fn active_model_for_returns_correct_model() {
        let coord = Coordinator::new(config_with_active_models());
        assert_eq!(
            coord.active_model_for(ModelCategory::Chat),
            Some("test-chat")
        );
        assert_eq!(
            coord.active_model_for(ModelCategory::Image),
            Some("test-image")
        );
        assert_eq!(coord.active_model_for(ModelCategory::Talk), None);
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
