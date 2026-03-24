use crate::device;
use crate::registry::find_manifest;
use crate::shared::types::ModelCategory;
use anyhow::Result;

impl super::Coordinator {
    pub fn load_model(&mut self, model_name: &str) -> Result<()> {
        if self.slots.get(model_name).is_some_and(|s| s.is_loaded()) {
            tracing::info!("model '{}' already loaded", model_name);
            return Ok(());
        }

        // Memory pre-flight check
        if let Some(manifest) = find_manifest(model_name) {
            let size_bytes = (manifest.manifest.size_gb * 1_000_000_000.0) as u64;
            device::preflight_memory_check(model_name, size_bytes)?;
        }

        if let Some(status) = device::memory_status_string() {
            tracing::info!("{status}");
        }

        let slot = self.create_model_slot(model_name)?;
        self.slots.insert(model_name.to_string(), slot);

        let slot = self
            .slots
            .get_mut(model_name)
            .ok_or_else(|| anyhow::anyhow!("model slot for '{}' disappeared", model_name))?;

        let load_start = std::time::Instant::now();
        slot.model.load()?;
        let load_time_ms = load_start.elapsed().as_millis() as u64;
        let memory_bytes = slot.memory_estimate_bytes();
        self.stats.record_model_loaded(model_name, load_time_ms, memory_bytes);

        tracing::info!("loaded model '{}' in {}ms", model_name, load_time_ms);
        Ok(())
    }

    pub fn load_defaults(&mut self, categories: &[ModelCategory]) -> Result<()> {
        for category in categories {
            if let Some(model_name) = self.config.default_for(*category) {
                let model_name = model_name.to_string();
                tracing::info!("loading default {} model: {}", category, model_name);
                self.load_model(&model_name)?;
            } else {
                tracing::warn!("no default model configured for {}", category);
            }
        }
        Ok(())
    }

    pub fn load_startup_models(&mut self) -> Result<()> {
        let categories: Vec<ModelCategory> = self
            .config
            .startup_categories
            .iter()
            .filter_map(|s| s.parse::<ModelCategory>().ok())
            .collect();
        self.load_defaults(&categories)
    }

    fn create_model_slot(&self, model_name: &str) -> Result<super::ModelSlot> {
        // Factory -- dispatches to model constructors based on manifest family.
        // Currently no models are implemented, so we return an error.
        anyhow::bail!(
            "model '{}' is not yet supported -- no model implementations available",
            model_name
        )
    }
}
