use crate::api::types::ModelCategory;
use crate::inference::candle::device;
use crate::registry::find_manifest;
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
        self.stats
            .record_model_loaded(model_name, load_time_ms, memory_bytes);

        tracing::info!("loaded model '{}' in {}ms", model_name, load_time_ms);
        Ok(())
    }

    pub fn load_active_models(&mut self, categories: &[ModelCategory]) -> Result<()> {
        for category in categories {
            if let Some(model_name) = self.config.active_model_for(*category) {
                let model_name = model_name.to_string();
                tracing::info!("loading active {} model: {}", category, model_name);
                self.load_model(&model_name)?;
            } else {
                tracing::warn!("no active model configured for {}", category);
            }
        }
        Ok(())
    }

    pub fn load_startup_categories(&mut self) -> Result<()> {
        let categories: Vec<ModelCategory> = self
            .config
            .startup_categories
            .iter()
            .filter_map(|s| s.parse::<ModelCategory>().ok())
            .collect();
        self.load_active_models(&categories)
    }

    fn create_model_slot(&mut self, model_name: &str) -> Result<super::ModelSlot> {
        let manifest = find_manifest(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown model '{model_name}'"))?;

        let model_dir = crate::download::paths::model_storage_dir(model_name);
        let categories = manifest.categories.clone();
        let memory_bytes = (manifest.manifest.size_gb * 1_000_000_000.0) as u64;

        #[cfg(feature = "mlx")]
        let mut factory =
            super::factory::ModelFactory::new(&self.config, &mut self.provider_servers);
        #[cfg(not(feature = "mlx"))]
        let mut factory = super::factory::ModelFactory::new(&self.config);

        let model = factory.build(
            manifest,
            model_name,
            model_dir,
            memory_bytes,
            categories.clone(),
        )?;

        Ok(super::ModelSlot::new(model, categories))
    }
}
