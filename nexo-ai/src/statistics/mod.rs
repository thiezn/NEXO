pub mod aggregates;
pub mod backend;
pub mod display;
pub mod metrics;

use backend::{InMemoryBackend, StatsBackend};
use metrics::{InferenceDetail, InferenceRecord, LifecycleEvent, LifecycleRecord};

use crate::shared::types::ModelCategory;
use std::time::SystemTime;

/// Central statistics facade. Owns a backend and provides
/// convenient recording methods that the coordinator calls.
pub struct StatsCollector {
    backend: Box<dyn StatsBackend>,
}

impl StatsCollector {
    pub fn new() -> Self {
        Self {
            backend: Box::new(InMemoryBackend::new(1000)),
        }
    }

    /// Create with a different backend (e.g. SQLite in the future).
    pub fn with_backend(backend: Box<dyn StatsBackend>) -> Self {
        Self { backend }
    }

    // ── Convenience recording methods ───────────────────────────────

    pub fn record_text_generation(
        &mut self,
        model_name: &str,
        category: ModelCategory,
        tokens_generated: usize,
        inference_time_ms: u64,
    ) {
        self.backend.record_inference(InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            category,
            inference_time_ms,
            detail: InferenceDetail::TextGeneration { tokens_generated },
        });
    }

    pub fn record_transcription(
        &mut self,
        model_name: &str,
        audio_duration_ms: u64,
        inference_time_ms: u64,
    ) {
        self.backend.record_inference(InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            category: ModelCategory::Listen,
            inference_time_ms,
            detail: InferenceDetail::Transcription { audio_duration_ms },
        });
    }

    pub fn record_synthesis(
        &mut self,
        model_name: &str,
        samples_generated: usize,
        sample_rate: u32,
        inference_time_ms: u64,
    ) {
        self.backend.record_inference(InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            category: ModelCategory::Talk,
            inference_time_ms,
            detail: InferenceDetail::Synthesis {
                samples_generated,
                sample_rate,
            },
        });
    }

    pub fn record_image_generation(
        &mut self,
        model_name: &str,
        images_generated: u32,
        steps: u32,
        total_pixels: u64,
        inference_time_ms: u64,
    ) {
        self.backend.record_inference(InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            category: ModelCategory::Imagine,
            inference_time_ms,
            detail: InferenceDetail::ImageGeneration {
                images_generated,
                steps,
                total_pixels,
            },
        });
    }

    pub fn record_model_loaded(&mut self, model_name: &str, load_time_ms: u64, memory_bytes: u64) {
        self.backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            event: LifecycleEvent::Loaded {
                load_time_ms,
                memory_bytes,
            },
        });
    }

    pub fn record_model_unloaded(&mut self, model_name: &str) {
        self.backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: model_name.to_string(),
            event: LifecycleEvent::Unloaded,
        });
    }

    // ── Query methods ───────────────────────────────────────────────

    pub fn all_stats(&self) -> Vec<&aggregates::ModelStats> {
        self.backend.all_stats()
    }

    pub fn model_stats(&self, name: &str, cat: ModelCategory) -> Option<&aggregates::ModelStats> {
        self.backend.model_stats(name, cat)
    }

    pub fn recent_inferences(&self, limit: usize) -> Vec<&InferenceRecord> {
        self.backend.recent_inferences(limit)
    }

    pub fn lifecycle_history(&self, model_name: &str) -> Vec<&LifecycleRecord> {
        self.backend.lifecycle_history(model_name)
    }

    pub fn clear(&mut self) {
        self.backend.clear();
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn record_and_query_text_generation() {
        let mut stats = StatsCollector::new();
        stats.record_text_generation("test-model", ModelCategory::Chat, 100, 500);
        stats.record_text_generation("test-model", ModelCategory::Chat, 200, 1000);

        let all = stats.all_stats();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].inference_count, 2);
        assert_eq!(all[0].total_inference_time_ms, 1500);
    }

    #[test]
    fn record_and_query_transcription() {
        let mut stats = StatsCollector::new();
        stats.record_transcription("whisper", 5000, 2000);

        let s = stats.model_stats("whisper", ModelCategory::Listen).unwrap();
        assert_eq!(s.inference_count, 1);
        // RTF = 2000 / 5000 = 0.4
        assert!((s.primary_metric.mean() - 0.4).abs() < 0.001);
    }

    #[test]
    fn record_and_query_synthesis() {
        let mut stats = StatsCollector::new();
        // 48000 samples at 24000 Hz = 2s of audio, produced in 1000ms
        stats.record_synthesis("tts-model", 48000, 24000, 1000);

        let s = stats.model_stats("tts-model", ModelCategory::Talk).unwrap();
        // x realtime = (48000 / 24000) / (1000 / 1000) = 2.0
        assert!((s.primary_metric.mean() - 2.0).abs() < 0.001);
    }

    #[test]
    fn record_and_query_image_generation() {
        let mut stats = StatsCollector::new();
        stats.record_image_generation("flux", 1, 20, 512 * 512, 4000);

        let s = stats.model_stats("flux", ModelCategory::Imagine).unwrap();
        // img/s = 1 / 4.0 = 0.25
        assert!((s.primary_metric.mean() - 0.25).abs() < 0.001);
    }

    #[test]
    fn record_lifecycle_events() {
        let mut stats = StatsCollector::new();
        stats.record_model_loaded("test", 1500, 4_000_000_000);
        stats.record_model_unloaded("test");

        let history = stats.lifecycle_history("test");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn clear_resets_everything() {
        let mut stats = StatsCollector::new();
        stats.record_text_generation("m", ModelCategory::Chat, 10, 100);
        stats.record_model_loaded("m", 500, 1000);
        assert!(!stats.all_stats().is_empty());

        stats.clear();
        assert!(stats.all_stats().is_empty());
        assert!(stats.lifecycle_history("m").is_empty());
    }
}
