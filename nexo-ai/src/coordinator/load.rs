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

        let model: Box<dyn crate::shared::model_traits::ModelInfo> =
            match manifest.manifest.family.as_str() {
                #[cfg(feature = "candle")]
                "whisper" => Box::new(crate::models::whisper::WhisperModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )),
                #[cfg(feature = "candle")]
                "flux" => Box::new(crate::models::flux2::FluxModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )),
                #[cfg(feature = "candle")]
                "gemma4" => {
                    let settings = self.config.model_settings(model_name);
                    let is_gguf = manifest
                        .manifest
                        .files
                        .iter()
                        .any(|f| f.hf_filename.ends_with(".gguf"));
                    Box::new(
                        crate::models::gemma4::Gemma4Model::new(
                            model_name.to_string(),
                            memory_bytes,
                            model_dir,
                        )
                        .with_gguf(is_gguf)
                        .with_max_context_tokens(settings.max_context_tokens),
                    )
                }
                #[cfg(feature = "candle")]
                "z_image" => Box::new(crate::models::z_image::ZImageModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )),
                #[cfg(feature = "candle")]
                "qwen_image" => Box::new(crate::models::qwen_image::QwenImageModel::new(
                    model_name.to_string(),
                    memory_bytes,
                    model_dir,
                )),
                #[cfg(feature = "mlx")]
                "mlx" => {
                    let hf_repo = manifest.hf_repo.clone().ok_or_else(|| {
                        anyhow::anyhow!("MLX model '{model_name}' missing hf_repo")
                    })?;
                    let (host, port) = self.config.mlx_server_addr();
                    let venv = self.config.mlx_venv_path.clone();
                    let server = self
                        .mlx_server
                        .get_or_insert_with(|| {
                            std::sync::Arc::new(tokio::sync::Mutex::new(
                                crate::remote_models::mlx_server::MlxServer::new(&host, port, venv),
                            ))
                        })
                        .clone();
                    let base_url = format!("http://{host}:{port}");
                    Box::new(crate::remote_models::mlx_model::MlxModel::new(
                        model_name,
                        &hf_repo,
                        model_dir,
                        memory_bytes,
                        categories.clone(),
                        server,
                        &base_url,
                    ))
                }
                other => anyhow::bail!("unsupported model family '{other}'"),
            };

        Ok(super::ModelSlot::new(model, categories))
    }
}
