pub mod candle;
pub mod common;
pub mod openai;

pub use common::config;

use std::path::PathBuf;

use anyhow::Result;

use crate::api::model_traits::{ImagineModel, ModelInfo};
use crate::api::types::{ImagineRequest, ImagineResponse, ModelCategory};

pub struct FluxModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<candle::pipeline::LoadedState>,
}

impl FluxModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

impl ModelInfo for FluxModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "flux"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Imagine]
    }

    fn memory_estimate_bytes(&self) -> u64 {
        self.memory_bytes
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }
        let variant = config::FluxVariant::from_name(&self.name);
        self.loaded = Some(candle::pipeline::load(&self.model_dir, variant)?);
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn as_imagine(&mut self) -> Option<&mut dyn ImagineModel> {
        Some(self)
    }
}

impl ImagineModel for FluxModel {
    fn imagine(&mut self, request: &ImagineRequest) -> Result<ImagineResponse> {
        let state = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        candle::pipeline::imagine(state, request)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_model() -> FluxModel {
        FluxModel::new(
            "flux-2-klein-4b".to_string(),
            12_000_000_000,
            PathBuf::from("/tmp/fake"),
        )
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "flux-2-klein-4b");
        assert_eq!(model.family(), "flux");
        assert_eq!(model.categories(), &[ModelCategory::Imagine]);
        assert_eq!(model.memory_estimate_bytes(), 12_000_000_000);
    }

    #[test]
    fn initially_not_loaded() {
        let model = make_model();
        assert!(!model.is_loaded());
    }

    #[test]
    fn as_imagine_returns_some() {
        let mut model = make_model();
        assert!(model.as_imagine().is_some());
    }

    #[test]
    fn other_downcasts_return_none() {
        let mut model = make_model();
        assert!(model.as_chat().is_none());
        assert!(model.as_tool().is_none());
        assert!(model.as_image().is_none());
        assert!(model.as_listen().is_none());
        assert!(model.as_talk().is_none());
        assert!(model.as_embed().is_none());
    }

    #[test]
    fn unload_when_not_loaded_is_noop() {
        let mut model = make_model();
        model.unload();
        assert!(!model.is_loaded());
    }

    #[test]
    fn imagine_errors_when_not_loaded() {
        let mut model = make_model();
        let request = ImagineRequest {
            prompt: "a sunset".into(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
            seed: 42,
            batch_size: 1,
        };
        let result = model.imagine(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }
}
