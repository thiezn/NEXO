pub mod gemma3_model;
pub mod pipeline;
pub mod template;
pub mod vision;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{ChatModel, ImageModel, ModelInfo, ToolModel};
use crate::shared::types::*;

pub struct Gemma3Model {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<pipeline::LoadedState>,
}

impl Gemma3Model {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

impl ModelInfo for Gemma3Model {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "gemma3"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image]
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

    fn as_chat(&mut self) -> Option<&mut dyn ChatModel> {
        Some(self)
    }

    fn as_tool(&mut self) -> Option<&mut dyn ToolModel> {
        Some(self)
    }

    fn as_image(&mut self) -> Option<&mut dyn ImageModel> {
        if self.loaded.as_ref().is_some_and(|s| s.vision.is_some()) {
            Some(self)
        } else {
            None
        }
    }
}

impl ChatModel for Gemma3Model {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        pipeline::chat(state, request)
    }
}

impl ToolModel for Gemma3Model {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        pipeline::call_tools(state, request)
    }
}

impl ImageModel for Gemma3Model {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let Some(vis) = state.vision.take() else {
            anyhow::bail!("vision components not loaded");
        };
        let result = vision::analyze_image(state, &vis, request);
        state.vision = Some(vis);
        result
    }
}
