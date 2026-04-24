#[cfg(feature = "candle")]
pub mod candle;
pub mod common;
#[cfg(feature = "mlx")]
pub mod openai;

#[cfg(feature = "candle")]
use std::path::PathBuf;

#[cfg(feature = "candle")]
use anyhow::Result;

#[cfg(feature = "candle")]
use crate::api::model_traits::{ListenModel, ModelInfo};
#[cfg(feature = "candle")]
use crate::api::types::{ListenRequest, ListenResponse, ModelCategory};

#[cfg(feature = "candle")]
pub struct WhisperModel {
    name: String,
    memory_bytes: u64,
    model_dir: PathBuf,
    loaded: Option<candle::pipeline::LoadedState>,
}

#[cfg(feature = "candle")]
impl WhisperModel {
    pub fn new(name: String, memory_bytes: u64, model_dir: PathBuf) -> Self {
        Self {
            name,
            memory_bytes,
            model_dir,
            loaded: None,
        }
    }
}

#[cfg(feature = "candle")]
impl ModelInfo for WhisperModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn family(&self) -> &str {
        "whisper"
    }

    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Listen]
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
        self.loaded = Some(candle::pipeline::load(&self.model_dir)?);
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn as_listen(&mut self) -> Option<&mut dyn ListenModel> {
        Some(self)
    }
}

#[cfg(feature = "candle")]
impl ListenModel for WhisperModel {
    fn transcribe(&mut self, request: &ListenRequest) -> Result<ListenResponse> {
        let state = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;
        candle::pipeline::transcribe(state, request)
    }
}

#[cfg(test)]
#[cfg(feature = "candle")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_model() -> WhisperModel {
        WhisperModel::new(
            "whisper-test".to_string(),
            1_500_000_000,
            PathBuf::from("/tmp/fake"),
        )
    }

    #[test]
    fn metadata() {
        let model = make_model();
        assert_eq!(model.name(), "whisper-test");
        assert_eq!(model.family(), "whisper");
        assert_eq!(model.categories(), &[ModelCategory::Listen]);
        assert_eq!(model.memory_estimate_bytes(), 1_500_000_000);
    }

    #[test]
    fn initially_not_loaded() {
        let model = make_model();
        assert!(!model.is_loaded());
    }

    #[test]
    fn as_listen_returns_some() {
        let mut model = make_model();
        assert!(model.as_listen().is_some());
    }

    #[test]
    fn other_downcasts_return_none() {
        let mut model = make_model();
        assert!(model.as_chat().is_none());
        assert!(model.as_tool().is_none());
        assert!(model.as_image().is_none());
        assert!(model.as_talk().is_none());
        assert!(model.as_imagine().is_none());
    }

    #[test]
    fn unload_when_not_loaded_is_noop() {
        let mut model = make_model();
        model.unload(); // should not panic
        assert!(!model.is_loaded());
    }

    #[test]
    fn transcribe_errors_when_not_loaded() {
        let mut model = make_model();
        let request = ListenRequest {
            pcm_samples: vec![0.0; 16000],
            sample_rate: 16000,
            language: None,
        };
        let result = model.transcribe(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }
}
