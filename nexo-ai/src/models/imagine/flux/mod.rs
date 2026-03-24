pub mod config;
pub mod pipeline;
pub(crate) mod sampling;
pub(crate) mod transformer;
pub(crate) mod vae;

use anyhow::Result;
use std::path::PathBuf;

use crate::shared::model_traits::{ImagineModel, ModelInfo};
use crate::shared::types::{ImagineRequest, ImagineResponse, ModelCategory};

pub struct FluxModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    variant: config::FluxVariant,
    loaded: Option<pipeline::LoadedState>,
}

impl FluxModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        let variant = config::FluxVariant::from_model_name(&name)
            .unwrap_or(config::FluxVariant::Klein4B);
        Self {
            name,
            memory_bytes,
            model_dir,
            variant,
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
        self.loaded = Some(pipeline::load(&self.model_dir, self.variant)?);
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
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::imagine(state, request)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn test_model() -> FluxModel {
        FluxModel::new(
            "flux-2-klein-4b".to_string(),
            22_000_000_000,
            PathBuf::from("/tmp/fake-flux"),
        )
    }

    #[test]
    fn metadata() {
        let m = test_model();
        assert_eq!(m.name(), "flux-2-klein-4b");
        assert_eq!(m.family(), "flux");
        assert_eq!(m.categories(), &[ModelCategory::Imagine]);
        assert_eq!(m.memory_estimate_bytes(), 22_000_000_000);
    }

    #[test]
    fn initially_not_loaded() {
        let m = test_model();
        assert!(!m.is_loaded());
    }

    #[test]
    fn unload_when_not_loaded_is_noop() {
        let mut m = test_model();
        m.unload(); // should not panic
        assert!(!m.is_loaded());
    }

    #[test]
    fn as_imagine_returns_some() {
        let mut m = test_model();
        assert!(m.as_imagine().is_some());
    }

    #[test]
    fn other_downcasts_return_none() {
        let mut m = test_model();
        assert!(m.as_chat().is_none());
        assert!(m.as_tool().is_none());
        assert!(m.as_image().is_none());
        assert!(m.as_listen().is_none());
        assert!(m.as_talk().is_none());
    }

    #[test]
    fn imagine_errors_when_not_loaded() {
        let mut m = test_model();
        let req = ImagineRequest {
            prompt: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 0.0,
            seed: 42,
            batch_size: 1,
        };
        let imagine = m.as_imagine().unwrap();
        let result = imagine.imagine(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn variant_detection() {
        let m4b = FluxModel::new("flux-2-klein-4b".into(), 0, PathBuf::new());
        assert_eq!(m4b.variant, config::FluxVariant::Klein4B);

        let m9b = FluxModel::new("flux-2-klein-9b".into(), 0, PathBuf::new());
        assert_eq!(m9b.variant, config::FluxVariant::Klein9B);

        let dev = FluxModel::new("flux-2-dev".into(), 0, PathBuf::new());
        assert_eq!(dev.variant, config::FluxVariant::Dev);
    }
}
