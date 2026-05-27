use crate::api::types::ModelCategory;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A single inference event, regardless of category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRecord {
    pub timestamp: SystemTime,
    pub model_name: String,
    pub category: ModelCategory,
    pub inference_time_ms: u64,
    pub detail: InferenceDetail,
}

/// Category-specific data needed to compute derived metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceDetail {
    /// Chat, Tool, or Image-analysis: token-generating models.
    TextGeneration {
        tokens_generated: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_tokens: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefix_reuse_tokens: Option<usize>,
    },
    /// Speech-to-text transcription.
    Transcription { audio_duration_ms: u64 },
    /// Text-to-speech synthesis.
    Synthesis {
        samples_generated: usize,
        sample_rate: u32,
    },
    /// Image generation.
    ImageGeneration {
        images_generated: u32,
        steps: u32,
        total_pixels: u64,
    },
}

impl InferenceDetail {
    /// Compute the primary performance metric for this inference.
    ///
    /// - TextGeneration: tokens per second
    /// - Transcription: real-time factor (lower is better, < 1.0 = faster than real-time)
    /// - Synthesis: x realtime (higher is better)
    /// - ImageGeneration: images per second
    pub fn primary_metric(&self, inference_time_ms: u64) -> f64 {
        let secs = inference_time_ms as f64 / 1000.0;
        if secs <= 0.0 {
            return 0.0;
        }

        match self {
            Self::TextGeneration {
                tokens_generated, ..
            } => *tokens_generated as f64 / secs,
            Self::Transcription { audio_duration_ms } => {
                if *audio_duration_ms == 0 {
                    return 0.0;
                }
                inference_time_ms as f64 / *audio_duration_ms as f64
            }
            Self::Synthesis {
                samples_generated,
                sample_rate,
            } => {
                if *sample_rate == 0 {
                    return 0.0;
                }
                let audio_secs = *samples_generated as f64 / *sample_rate as f64;
                audio_secs / secs
            }
            Self::ImageGeneration {
                images_generated, ..
            } => *images_generated as f64 / secs,
        }
    }

    /// Compute the secondary performance metric, if applicable.
    ///
    /// - ImageGeneration: steps per second
    /// - All others: None
    pub fn secondary_metric(&self, inference_time_ms: u64) -> Option<f64> {
        let secs = inference_time_ms as f64 / 1000.0;
        if secs <= 0.0 {
            return None;
        }

        match self {
            Self::ImageGeneration {
                steps,
                images_generated,
                ..
            } => Some((*steps as f64 * *images_generated as f64) / secs),
            _ => None,
        }
    }

    pub fn primary_metric_label(&self) -> &'static str {
        match self {
            Self::TextGeneration { .. } => "tok/s",
            Self::Transcription { .. } => "RTF",
            Self::Synthesis { .. } => "x realtime",
            Self::ImageGeneration { .. } => "img/s",
        }
    }

    pub fn secondary_metric_label(&self) -> Option<&'static str> {
        match self {
            Self::ImageGeneration { .. } => Some("step/s"),
            _ => None,
        }
    }
}

/// Record of a model load or unload event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRecord {
    pub timestamp: SystemTime,
    pub model_name: String,
    pub event: LifecycleEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleEvent {
    Loaded {
        load_time_ms: u64,
        memory_bytes: u64,
    },
    Unloaded,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn text_generation_metric() {
        let detail = InferenceDetail::TextGeneration {
            tokens_generated: 100,
            prompt_tokens: None,
            prefix_reuse_tokens: None,
        };
        let metric = detail.primary_metric(500);
        assert!((metric - 200.0).abs() < 0.01); // 100 tokens / 0.5s = 200 tok/s
        assert_eq!(detail.primary_metric_label(), "tok/s");
        assert!(detail.secondary_metric(500).is_none());
    }

    #[test]
    fn transcription_metric() {
        let detail = InferenceDetail::Transcription {
            audio_duration_ms: 10000,
        };
        let metric = detail.primary_metric(3000);
        assert!((metric - 0.3).abs() < 0.01); // 3000ms / 10000ms = 0.3 RTF
        assert_eq!(detail.primary_metric_label(), "RTF");
    }

    #[test]
    fn synthesis_metric() {
        let detail = InferenceDetail::Synthesis {
            samples_generated: 48000,
            sample_rate: 24000,
        };
        let metric = detail.primary_metric(1000);
        assert!((metric - 2.0).abs() < 0.01); // 2s audio / 1s inference = 2x realtime
        assert_eq!(detail.primary_metric_label(), "x realtime");
    }

    #[test]
    fn image_generation_metrics() {
        let detail = InferenceDetail::ImageGeneration {
            images_generated: 2,
            steps: 20,
            total_pixels: 512 * 512 * 2,
        };
        let primary = detail.primary_metric(4000);
        assert!((primary - 0.5).abs() < 0.01); // 2 images / 4s = 0.5 img/s

        let secondary = detail.secondary_metric(4000).unwrap();
        assert!((secondary - 10.0).abs() < 0.01); // (20 * 2) / 4s = 10 step/s

        assert_eq!(detail.primary_metric_label(), "img/s");
        assert_eq!(detail.secondary_metric_label(), Some("step/s"));
    }

    #[test]
    fn zero_time_returns_zero() {
        let detail = InferenceDetail::TextGeneration {
            tokens_generated: 100,
            prompt_tokens: None,
            prefix_reuse_tokens: None,
        };
        assert_eq!(detail.primary_metric(0), 0.0);
    }
}
