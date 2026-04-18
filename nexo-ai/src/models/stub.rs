use crate::shared::model_traits::*;
use crate::shared::types::*;
use anyhow::Result;

pub struct StubModel {
    name: String,
    loaded: bool,
    memory_bytes: u64,
}

impl StubModel {
    pub fn new(name: &str, memory_bytes: u64) -> Self {
        Self {
            name: name.to_string(),
            loaded: false,
            memory_bytes,
        }
    }
}

impl ModelInfo for StubModel {
    fn name(&self) -> &str {
        &self.name
    }
    fn family(&self) -> &str {
        "stub"
    }
    fn categories(&self) -> &[ModelCategory] {
        &[ModelCategory::Chat]
    }
    fn memory_estimate_bytes(&self) -> u64 {
        self.memory_bytes
    }
    fn is_loaded(&self) -> bool {
        self.loaded
    }
    fn load(&mut self) -> Result<()> {
        self.loaded = true;
        Ok(())
    }
    fn unload(&mut self) {
        self.loaded = false;
    }
    fn as_chat(&mut self) -> Option<&mut dyn ChatModel> {
        Some(self)
    }
}

impl ChatModel for StubModel {
    fn chat(&mut self, _request: &ChatRequest) -> Result<ChatResponse> {
        if !self.loaded {
            anyhow::bail!("model not loaded");
        }
        Ok(ChatResponse {
            text: "stub response".to_string(),
            tokens_generated: 2,
            inference_time_ms: 0,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_new_load_unload() {
        let mut model = StubModel::new("test-stub", 1_000_000);
        assert!(!model.is_loaded());
        assert_eq!(model.name(), "test-stub");
        assert_eq!(model.family(), "stub");
        assert_eq!(model.memory_estimate_bytes(), 1_000_000);

        model.load().unwrap();
        assert!(model.is_loaded());

        model.unload();
        assert!(!model.is_loaded());
    }

    #[test]
    fn chat_errors_when_not_loaded() {
        let mut model = StubModel::new("unloaded", 500);
        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };
        let result = model.chat(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model not loaded"));
    }

    #[test]
    fn chat_returns_response_when_loaded() {
        let mut model = StubModel::new("loaded", 500);
        model.load().unwrap();

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            session_id: None,
        };
        let response = model.chat(&request).unwrap();
        assert_eq!(response.text, "stub response");
        assert_eq!(response.tokens_generated, 2);
        assert_eq!(response.inference_time_ms, 0);
    }
}
