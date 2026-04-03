pub mod model;
pub mod pipeline;
pub mod template;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{ChatModel, ImageModel, KvCacheable, ModelInfo, ToolModel};
use crate::shared::types::{
    ChatRequest, ChatResponse, ImageAnalysisRequest, ImageAnalysisResponse, LayerKvSnapshot,
    ModelCategory, ToolCallRequest, ToolCallResponse,
};

pub struct Gemma4Model {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<pipeline::LoadedState>,
    max_context_tokens: Option<usize>,
}

impl Gemma4Model {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
            max_context_tokens: None,
        }
    }

    #[must_use]
    pub fn with_max_context_tokens(mut self, max_context_tokens: Option<usize>) -> Self {
        self.max_context_tokens = max_context_tokens;
        self
    }
}

impl ModelInfo for Gemma4Model {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "gemma4"
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
        self.loaded = Some(pipeline::load(&self.model_dir, self.max_context_tokens)?);
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
        Some(self)
    }

    fn as_kv_cacheable(&mut self) -> Option<&mut dyn KvCacheable> {
        Some(self)
    }
}

impl ChatModel for Gemma4Model {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::chat(state, request)
    }
}

impl ToolModel for Gemma4Model {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::call_tools(state, request)
    }
}

impl KvCacheable for Gemma4Model {
    fn kv_cache_seq_len(&self) -> usize {
        self.loaded.as_ref().map(|s| s.kv_cache_seq_len()).unwrap_or(0)
    }

    fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>> {
        let state = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        state.save_kv_cache().map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        state.restore_kv_cache(snapshots).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn clear_kv_cache(&mut self) {
        if let Some(state) = self.loaded.as_mut() {
            state.clear_kv_cache();
        }
    }

    fn processed_tokens(&self) -> &[u32] {
        self.loaded
            .as_ref()
            .map(|s| s.processed_tokens())
            .unwrap_or(&[])
    }

    fn current_session_id(&self) -> Option<&str> {
        self.loaded.as_ref().and_then(|s| s.current_session_id())
    }

    fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>) {
        if let Some(state) = self.loaded.as_mut() {
            state.set_session_state(session_id, tokens);
        }
    }

    fn device(&self) -> &candle_core::Device {
        self.loaded
            .as_ref()
            .expect("model must be loaded to access device")
            .device()
    }

    fn dtype(&self) -> candle_core::DType {
        self.loaded
            .as_ref()
            .expect("model must be loaded to access dtype")
            .dtype()
    }
}

impl ImageModel for Gemma4Model {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::analyze_image(state, request, &self.model_dir)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_model() -> Gemma4Model {
        Gemma4Model::new(
            "gemma-4-e4b-it".to_string(),
            14_890_000_000,
            PathBuf::from("/tmp/fake"),
        )
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "gemma-4-e4b-it");
        assert_eq!(model.family(), "gemma4");
        assert_eq!(
            model.categories(),
            &[ModelCategory::Chat, ModelCategory::Tool, ModelCategory::Image]
        );
        assert_eq!(model.memory_estimate_bytes(), 14_890_000_000);
    }

    #[test]
    fn initially_not_loaded() {
        let model = make_model();
        assert!(!model.is_loaded());
    }

    #[test]
    fn as_chat_returns_some() {
        let mut model = make_model();
        assert!(model.as_chat().is_some());
    }

    #[test]
    fn as_tool_returns_some() {
        let mut model = make_model();
        assert!(model.as_tool().is_some());
    }

    #[test]
    fn as_image_returns_some() {
        let mut model = make_model();
        assert!(model.as_image().is_some());
    }

    #[test]
    fn as_kv_cacheable_returns_some() {
        let mut model = make_model();
        assert!(model.as_kv_cacheable().is_some());
    }

    #[test]
    fn other_downcasts_return_none() {
        let mut model = make_model();
        assert!(model.as_listen().is_none());
        assert!(model.as_talk().is_none());
        assert!(model.as_imagine().is_none());
        assert!(model.as_embed().is_none());
    }

    #[test]
    fn unload_when_not_loaded_is_noop() {
        let mut model = make_model();
        model.unload();
        assert!(!model.is_loaded());
    }

    #[test]
    fn chat_errors_when_not_loaded() {
        let mut model = make_model();
        let request = ChatRequest {
            messages: vec![],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };
        let result = model.chat(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn call_tools_errors_when_not_loaded() {
        let mut model = make_model();
        let request = ToolCallRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };
        let result = model.call_tools(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn analyze_image_errors_when_not_loaded() {
        let mut model = make_model();
        let request = ImageAnalysisRequest {
            image_data: vec![0u8; 100],
            prompt: "describe this".into(),
            max_tokens: 100,
            temperature: 0.7,
        };
        let result = model.analyze_image(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn with_max_context_tokens() {
        let model = make_model().with_max_context_tokens(Some(4096));
        assert_eq!(model.max_context_tokens, Some(4096));
    }
}
