//! Z-Image model family (text-to-image generation).
//!
//! GGUF-based inference pipeline — TODO: implement using quantized weights.

use anyhow::Result;
use std::path::PathBuf;

use crate::shared::model_traits::{ImagineModel, ModelInfo};
use crate::shared::types::{ImagineRequest, ImagineResponse, ModelCategory};

pub struct ZImageModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: bool,
}

impl ZImageModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: false,
        }
    }
}

impl ModelInfo for ZImageModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "z_image"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Imagine]
    }

    fn memory_estimate_bytes(&self) -> u64 {
        self.memory_bytes
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        anyhow::ensure!(
            self.model_dir.exists(),
            "model directory not found: {}",
            self.model_dir.display()
        );
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = false;
    }

    fn as_imagine(&mut self) -> Option<&mut dyn ImagineModel> {
        Some(self)
    }
}

impl ImagineModel for ZImageModel {
    fn imagine(&mut self, _request: &ImagineRequest) -> Result<ImagineResponse> {
        anyhow::bail!("Z-Image GGUF inference not yet implemented")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn model_metadata() {
        let model = ZImageModel::new("z-image-turbo".into(), 30_000_000_000, PathBuf::from("/tmp"));
        assert_eq!(model.name(), "z-image-turbo");
        assert_eq!(model.family(), "z_image");
        assert_eq!(model.categories(), &[ModelCategory::Imagine]);
        assert!(!model.is_loaded());
    }

    #[test]
    fn downcast_returns_imagine() {
        let mut model = ZImageModel::new("test".into(), 0, PathBuf::from("/tmp"));
        assert!(model.as_imagine().is_some());
        assert!(model.as_chat().is_none());
        assert!(model.as_tool().is_none());
        assert!(model.as_image().is_none());
        assert!(model.as_listen().is_none());
        assert!(model.as_talk().is_none());
        assert!(model.as_embed().is_none());
    }
}
