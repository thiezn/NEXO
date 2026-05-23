use serde::{Deserialize, Serialize};

/// Per-model runtime overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelSettings {
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    pub default_steps: Option<u32>,
    pub default_guidance: Option<f64>,
    pub default_width: Option<u32>,
    pub default_height: Option<u32>,
    pub voice_description: Option<String>,
    /// Maximum number of tokens (prompt + generation) allowed in the KV cache.
    /// When set, generation will fail if the prompt exceeds this budget,
    /// signalling the caller to slide the conversation window.
    pub max_context_tokens: Option<usize>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn model_settings_serde_roundtrip() {
        let settings = ModelSettings {
            temperature: Some(0.5),
            max_tokens: Some(1024),
            top_p: Some(0.95),
            top_k: Some(64),
            seed: Some(123),
            default_steps: Some(20),
            default_guidance: Some(7.5),
            default_width: Some(512),
            default_height: Some(512),
            voice_description: Some("warm male voice".to_string()),
            max_context_tokens: Some(8192),
        };
        let toml_str = toml::to_string(&settings).unwrap();
        let parsed: ModelSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.temperature, Some(0.5));
        assert_eq!(parsed.max_tokens, Some(1024));
        assert_eq!(parsed.seed, Some(123));
        assert_eq!(parsed.default_steps, Some(20));
        assert_eq!(parsed.default_guidance, Some(7.5));
        assert_eq!(
            parsed.voice_description,
            Some("warm male voice".to_string())
        );
    }
}
