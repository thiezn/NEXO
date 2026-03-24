pub mod config;
pub mod pipeline;

use std::path::PathBuf;

use anyhow::Result;

use crate::shared::model_traits::{ModelInfo, TalkModel};
use crate::shared::types::{ModelCategory, TalkRequest, TalkResponse};

pub struct ParlerTtsModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<pipeline::LoadedState>,
}

impl ParlerTtsModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

impl ModelInfo for ParlerTtsModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "parler"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Talk]
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

    fn as_talk(&mut self) -> Option<&mut dyn TalkModel> {
        Some(self)
    }
}

impl TalkModel for ParlerTtsModel {
    fn synthesize(&mut self, request: &TalkRequest) -> Result<TalkResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        pipeline::synthesize(state, request)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_model() -> ParlerTtsModel {
        ParlerTtsModel::new("parler-test".to_string(), 4_000_000_000, PathBuf::from("/tmp/fake"))
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "parler-test");
        assert_eq!(model.family(), "parler");
        assert_eq!(model.categories(), &[ModelCategory::Talk]);
        assert_eq!(model.memory_estimate_bytes(), 4_000_000_000);
    }

    #[test]
    fn initially_not_loaded() {
        let model = make_model();
        assert!(!model.is_loaded());
    }

    #[test]
    fn as_talk_returns_some() {
        let mut model = make_model();
        assert!(model.as_talk().is_some());
    }

    #[test]
    fn other_downcasts_return_none() {
        let mut model = make_model();
        assert!(model.as_chat().is_none());
        assert!(model.as_tool().is_none());
        assert!(model.as_image().is_none());
        assert!(model.as_listen().is_none());
        assert!(model.as_imagine().is_none());
    }

    #[test]
    fn unload_when_not_loaded_is_noop() {
        let mut model = make_model();
        model.unload(); // should not panic
        assert!(!model.is_loaded());
    }

    #[test]
    fn synthesize_errors_when_not_loaded() {
        let mut model = make_model();
        let request = TalkRequest {
            text: "hello".to_string(),
            voice_description: "warm".to_string(),
            max_tokens: 100,
            temperature: 1.0,
            seed: 0,
        };
        let result = model.synthesize(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }
}
