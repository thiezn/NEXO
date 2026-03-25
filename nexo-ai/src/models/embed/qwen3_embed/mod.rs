pub mod pipeline;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{EmbedModel, ModelInfo};
use crate::shared::types::*;

pub struct Qwen3EmbedModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<pipeline::LoadedState>,
}

impl Qwen3EmbedModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

impl ModelInfo for Qwen3EmbedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "qwen3_embed"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Embed]
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

    fn as_embed(&mut self) -> Option<&mut dyn EmbedModel> {
        Some(self)
    }
}

impl EmbedModel for Qwen3EmbedModel {
    fn embed(&mut self, request: &EmbedRequest) -> Result<EmbedResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        pipeline::embed(state, request)
    }
}
