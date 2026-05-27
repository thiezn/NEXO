use std::collections::HashMap;

use crate::api::model_traits::ModelInfo;
use crate::api::types::ModelCategory;
use crate::config::CoordinatorConfig;
use crate::statistics::StatsCollector;

#[cfg(feature = "mlx")]
use crate::inference::remote::servers::ManagedProviderServers;

pub struct ModelSlot {
    pub(super) model: Box<dyn ModelInfo>,
    pub(super) categories: Vec<ModelCategory>,
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
    pub(super) config: CoordinatorConfig,
    pub(super) slots: HashMap<String, ModelSlot>,
    pub(super) active_models: HashMap<ModelCategory, String>,
    pub(super) stats: StatsCollector,
    #[cfg(feature = "mlx")]
    pub(super) provider_servers: ManagedProviderServers,
}

impl Coordinator {
    pub fn new(config: CoordinatorConfig) -> Self {
        let active_models = config
            .active_models
            .iter()
            .filter_map(|(cat_str, model_name)| {
                cat_str
                    .parse::<ModelCategory>()
                    .ok()
                    .map(|category| (category, model_name.clone()))
            })
            .collect();

        Self {
            config,
            slots: HashMap::new(),
            active_models,
            stats: StatsCollector::new(),
            #[cfg(feature = "mlx")]
            provider_servers: ManagedProviderServers::default(),
        }
    }

    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut CoordinatorConfig {
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
            .filter(|slot| slot.is_loaded())
            .map(|slot| (slot.name(), slot.categories()))
            .collect()
    }

    pub fn loaded_model_count(&self) -> usize {
        self.slots.values().filter(|slot| slot.is_loaded()).count()
    }

    pub fn active_model_for(&self, category: ModelCategory) -> Option<&str> {
        self.active_models.get(&category).map(String::as_str)
    }

    pub fn set_active_model(&mut self, category: ModelCategory, model_name: String) {
        self.active_models.insert(category, model_name.clone());
        self.config.set_active_model(category, model_name);
    }

    pub fn remove_active_model(&mut self, category: ModelCategory) {
        self.active_models.remove(&category);
        self.config.remove_active_model(category);
    }

    pub fn clear_active_models(&mut self) {
        self.active_models.clear();
        self.config.clear_active_models();
    }

    pub fn is_model_loaded(&self, name: &str) -> bool {
        self.slots.get(name).is_some_and(|slot| slot.is_loaded())
    }

    pub fn list_models(&self) -> Vec<crate::registry::ModelEntry> {
        crate::registry::list_models(|name| self.is_model_loaded(name))
    }

    pub fn model_mut(&mut self, name: &str) -> Option<&mut dyn crate::api::model_traits::ModelInfo> {
        self.slots.get_mut(name).map(ModelSlot::model_mut)
    }

    pub fn total_memory_used(&self) -> u64 {
        self.slots
            .values()
            .filter(|slot| slot.is_loaded())
            .map(|slot| slot.memory_estimate_bytes())
            .sum()
    }
}

#[cfg(test)]
#[cfg(feature = "candle")]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::inference::models::stub::StubModel;

    fn config_with_active_models() -> CoordinatorConfig {
        let mut config = CoordinatorConfig::default();
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
        let coord = Coordinator::new(CoordinatorConfig::default());
        assert!(coord.loaded_models().is_empty());
    }

    #[test]
    fn total_memory_used_zero_when_nothing_loaded() {
        let coord = Coordinator::new(CoordinatorConfig::default());
        assert_eq!(coord.total_memory_used(), 0);
    }

    #[test]
    fn unload_all_clears_slots() {
        let mut coord = Coordinator::new(CoordinatorConfig::default());
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
        let mut coord = Coordinator::new(CoordinatorConfig::default());

        let mut small = StubModel::new("small", 1_000_000);
        small.load().unwrap();
        coord.slots.insert(
            "small".to_string(),
            ModelSlot::new(Box::new(small), vec![ModelCategory::Chat]),
        );

        let mut large = StubModel::new("large", 5_000_000);
        large.load().unwrap();
        coord.slots.insert(
            "large".to_string(),
            ModelSlot::new(Box::new(large), vec![ModelCategory::Image]),
        );

        assert_eq!(coord.total_memory_used(), 6_000_000);

        let freed = coord.free_memory(4_000_000).unwrap();
        assert!(freed >= 4_000_000);
    }
}