pub mod config;
pub mod generation;
pub mod gguf;
pub mod safetensors;
pub mod template;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{
    AudioAnalysisModel, ChatModel, ImageModel, KvCacheable, ModelInfo, ToolModel,
};
use crate::shared::types::{
    AudioAnalysisRequest, AudioAnalysisResponse, ChatRequest, ChatResponse, ImageAnalysisRequest,
    ImageAnalysisResponse, LayerKvSnapshot, ModelCategory, ToolCallRequest, ToolCallResponse,
};

// ── Loaded variant ────────────────────────────────────────────────────────

enum LoadedVariant {
    Safetensors(safetensors::pipeline::LoadedState),
    Gguf(gguf::pipeline::LoadedState),
}

/// Macro to dispatch a method call to the inner variant, reducing match-arm boilerplate.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            LoadedVariant::Safetensors(s) => s.$method($($arg),*),
            LoadedVariant::Gguf(s) => s.$method($($arg),*),
        }
    };
}

// ── Categories ────────────────────────────────────────────────────────────

const CATEGORIES_ALL: &[ModelCategory] = &[
    ModelCategory::Chat,
    ModelCategory::Tool,
    ModelCategory::Image,
    ModelCategory::Listen,
];

const CATEGORIES_CHAT_TOOL_IMAGE: &[ModelCategory] = &[
    ModelCategory::Chat,
    ModelCategory::Tool,
    ModelCategory::Image,
];

/// Determine categories for a Gemma 4 model based on its name.
/// - E2B/E4B/31B (safetensors & GGUF): Chat, Tool, Image, Listen
/// - 26B-A4B: Chat, Tool, Image (no audio tower)
fn model_categories(name: &str) -> &'static [ModelCategory] {
    if name.contains("26b") {
        CATEGORIES_CHAT_TOOL_IMAGE
    } else {
        CATEGORIES_ALL
    }
}

// ── Gemma4Model ───────────────────────────────────────────────────────────

pub struct Gemma4Model {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<LoadedVariant>,
    max_context_tokens: Option<usize>,
    is_gguf: bool,
}

impl Gemma4Model {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
            max_context_tokens: None,
            is_gguf: false,
        }
    }

    #[must_use]
    pub fn with_max_context_tokens(mut self, max_context_tokens: Option<usize>) -> Self {
        self.max_context_tokens = max_context_tokens;
        self
    }

    #[must_use]
    pub fn with_gguf(mut self, is_gguf: bool) -> Self {
        self.is_gguf = is_gguf;
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
        model_categories(&self.name)
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
        if self.is_gguf {
            self.loaded = Some(LoadedVariant::Gguf(gguf::pipeline::load(
                &self.model_dir,
                self.max_context_tokens,
            )?));
        } else {
            self.loaded = Some(LoadedVariant::Safetensors(safetensors::pipeline::load(
                &self.model_dir,
                self.max_context_tokens,
            )?));
        }
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
        if self.categories().contains(&ModelCategory::Image) {
            Some(self)
        } else {
            None
        }
    }

    fn as_audio_analysis(&mut self) -> Option<&mut dyn AudioAnalysisModel> {
        if self.categories().contains(&ModelCategory::Listen) {
            Some(self)
        } else {
            None
        }
    }

    fn as_kv_cacheable(&mut self) -> Option<&mut dyn KvCacheable> {
        Some(self)
    }
}

impl ChatModel for Gemma4Model {
    fn chat(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        match self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?
        {
            LoadedVariant::Safetensors(state) => safetensors::pipeline::chat(state, request),
            LoadedVariant::Gguf(state) => gguf::pipeline::chat(state, request),
        }
    }
}

impl ToolModel for Gemma4Model {
    fn call_tools(&mut self, request: &ToolCallRequest) -> Result<ToolCallResponse> {
        match self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?
        {
            LoadedVariant::Safetensors(state) => safetensors::pipeline::call_tools(state, request),
            LoadedVariant::Gguf(state) => gguf::pipeline::call_tools(state, request),
        }
    }
}

impl KvCacheable for Gemma4Model {
    fn kv_cache_seq_len(&self) -> usize {
        self.loaded
            .as_ref()
            .map(|v| dispatch!(v, kv_cache_seq_len))
            .unwrap_or(0)
    }

    fn save_kv_cache(&self) -> Result<Vec<LayerKvSnapshot>> {
        let v = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        dispatch!(v, save_kv_cache).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> Result<()> {
        let v = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        dispatch!(v, restore_kv_cache, snapshots).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn clear_kv_cache(&mut self) {
        if let Some(v) = &mut self.loaded {
            dispatch!(v, clear_kv_cache);
        }
    }

    fn processed_tokens(&self) -> &[u32] {
        self.loaded
            .as_ref()
            .map(|v| dispatch!(v, processed_tokens))
            .unwrap_or(&[])
    }

    fn current_session_id(&self) -> Option<&str> {
        self.loaded
            .as_ref()
            .and_then(|v| dispatch!(v, current_session_id))
    }

    fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>) {
        if let Some(v) = &mut self.loaded {
            dispatch!(v, set_session_state, session_id, tokens);
        }
    }

    fn device(&self) -> &candle_core::Device {
        dispatch!(
            self.loaded
                .as_ref()
                .expect("model must be loaded to access device"),
            device
        )
    }

    fn dtype(&self) -> candle_core::DType {
        dispatch!(
            self.loaded
                .as_ref()
                .expect("model must be loaded to access dtype"),
            dtype
        )
    }
}

impl ImageModel for Gemma4Model {
    fn analyze_image(&mut self, request: &ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        let model_dir = self.model_dir.clone();
        match self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?
        {
            LoadedVariant::Safetensors(state) => {
                safetensors::pipeline::analyze_image(state, request, &model_dir)
            }
            LoadedVariant::Gguf(state) => gguf::pipeline::analyze_image(state, request, &model_dir),
        }
    }
}

impl AudioAnalysisModel for Gemma4Model {
    fn analyze_audio(&mut self, request: &AudioAnalysisRequest) -> Result<AudioAnalysisResponse> {
        let model_dir = self.model_dir.clone();
        match self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?
        {
            LoadedVariant::Safetensors(state) => {
                safetensors::pipeline::analyze_audio(state, request, &model_dir)
            }
            LoadedVariant::Gguf(state) => gguf::pipeline::analyze_audio(state, request, &model_dir),
        }
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

    fn make_gguf_model() -> Gemma4Model {
        Gemma4Model::new(
            "gemma-4-e2b-it-q5".to_string(),
            3_400_000_000,
            PathBuf::from("/tmp/fake"),
        )
        .with_gguf(true)
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "gemma-4-e4b-it");
        assert_eq!(model.family(), "gemma4");
        assert_eq!(
            model.categories(),
            &[
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image,
                ModelCategory::Listen,
            ]
        );
        assert_eq!(model.memory_estimate_bytes(), 14_890_000_000);
    }

    #[test]
    fn gguf_categories() {
        let model = make_gguf_model();
        assert_eq!(
            model.categories(),
            &[
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image,
                ModelCategory::Listen,
            ]
        );
    }

    #[test]
    fn gguf_26b_categories_no_audio() {
        let model = Gemma4Model::new(
            "gemma-4-26b-a4b-it-q4".to_string(),
            16_800_000_000,
            PathBuf::from("/tmp/fake"),
        )
        .with_gguf(true);
        assert_eq!(
            model.categories(),
            &[
                ModelCategory::Chat,
                ModelCategory::Tool,
                ModelCategory::Image
            ]
        );
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
    fn gguf_as_image_returns_some() {
        let mut model = make_gguf_model();
        assert!(model.as_image().is_some());
    }

    #[test]
    fn gguf_as_audio_analysis_returns_some() {
        let mut model = make_gguf_model();
        assert!(model.as_audio_analysis().is_some());
    }

    #[test]
    fn gguf_26b_as_audio_analysis_returns_none() {
        let mut model = Gemma4Model::new(
            "gemma-4-26b-a4b-it-q4".to_string(),
            16_800_000_000,
            PathBuf::from("/tmp/fake"),
        )
        .with_gguf(true);
        assert!(model.as_audio_analysis().is_none());
        assert!(model.as_image().is_some());
    }

    #[test]
    fn as_kv_cacheable_returns_some() {
        let mut model = make_model();
        assert!(model.as_kv_cacheable().is_some());
    }

    #[test]
    fn as_audio_analysis_returns_some() {
        let mut model = make_model();
        assert!(model.as_audio_analysis().is_some());
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
