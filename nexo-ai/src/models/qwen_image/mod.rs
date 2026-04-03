mod pipeline;
pub(crate) mod quantized_transformer;
pub(crate) mod sampling;
pub(crate) mod transformer;
mod vae;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{ImagineModel, ModelInfo};
use crate::shared::types::{ImagineRequest, ImagineResponse, ModelCategory};

pub struct QwenImageModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<pipeline::LoadedState>,
}

impl QwenImageModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

impl ModelInfo for QwenImageModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "qwen_image"
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
        self.loaded = Some(pipeline::load(&self.model_dir)?);
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn as_imagine(&mut self) -> Option<&mut dyn ImagineModel> {
        Some(self)
    }
}

impl ImagineModel for QwenImageModel {
    fn imagine(&mut self, request: &ImagineRequest) -> Result<ImagineResponse> {
        let state = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::generate(state, &self.model_dir, request)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_model() -> QwenImageModel {
        QwenImageModel::new(
            "qwen-image-test".to_string(),
            10_000_000_000,
            PathBuf::from("/tmp/fake"),
        )
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "qwen-image-test");
        assert_eq!(model.family(), "qwen_image");
        assert_eq!(model.categories(), &[ModelCategory::Imagine]);
        assert_eq!(model.memory_estimate_bytes(), 10_000_000_000);
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
            prompt: "a cat".into(),
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 7.5,
            seed: 42,
            batch_size: 1,
        };
        let result = model.imagine(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }
}
