use super::aggregates::{ModelStats, RunningStat};
use super::metrics::{InferenceRecord, LifecycleRecord};
use crate::api::types::ModelCategory;
use std::collections::{HashMap, VecDeque};

/// Abstraction over statistics storage.
/// In-memory now; file-based or SQLite later.
pub trait StatsBackend: Send {
    fn record_inference(&mut self, record: InferenceRecord);
    fn record_lifecycle(&mut self, record: LifecycleRecord);
    fn model_stats(&self, model_name: &str, category: ModelCategory) -> Option<&ModelStats>;
    fn all_stats(&self) -> Vec<&ModelStats>;
    fn recent_inferences(&self, limit: usize) -> Vec<&InferenceRecord>;
    fn lifecycle_history(&self, model_name: &str) -> Vec<&LifecycleRecord>;
    fn clear(&mut self);
}

/// In-memory statistics backend.
/// Stores the last `max_history` individual records and maintains
/// running aggregates for each (model, category) pair.
pub struct InMemoryBackend {
    inferences: VecDeque<InferenceRecord>,
    lifecycles: Vec<LifecycleRecord>,
    /// Two-level map: model_name → (category → stats).
    /// Avoids cloning the model name string on every lookup.
    aggregates: HashMap<String, HashMap<ModelCategory, ModelStats>>,
    max_history: usize,
}

impl InMemoryBackend {
    pub fn new(max_history: usize) -> Self {
        Self {
            inferences: VecDeque::with_capacity(max_history.min(1024)),
            lifecycles: Vec::new(),
            aggregates: HashMap::new(),
            max_history,
        }
    }
}

impl StatsBackend for InMemoryBackend {
    fn record_inference(&mut self, record: InferenceRecord) {
        let primary = record.detail.primary_metric(record.inference_time_ms);
        let secondary = record.detail.secondary_metric(record.inference_time_ms);

        let by_category = self
            .aggregates
            .entry(record.model_name.clone())
            .or_default();

        let stats = by_category.entry(record.category).or_insert_with(|| {
            let mut s = ModelStats::new(
                record.model_name.clone(),
                record.category,
                record.detail.primary_metric_label(),
            );
            if let Some(label) = record.detail.secondary_metric_label() {
                s.secondary_metric = Some(RunningStat::new());
                s.secondary_metric_label = Some(label.to_string());
            }
            s
        });

        stats.inference_count += 1;
        stats.total_inference_time_ms += record.inference_time_ms;
        stats.primary_metric.push(primary);

        if let (Some(sec_value), Some(sec_stat)) = (secondary, stats.secondary_metric.as_mut()) {
            sec_stat.push(sec_value);
        }

        // Ring buffer: drop oldest if full.
        if self.inferences.len() >= self.max_history {
            self.inferences.pop_front();
        }
        self.inferences.push_back(record);
    }

    fn record_lifecycle(&mut self, record: LifecycleRecord) {
        self.lifecycles.push(record);
    }

    fn model_stats(&self, model_name: &str, category: ModelCategory) -> Option<&ModelStats> {
        self.aggregates.get(model_name)?.get(&category)
    }

    fn all_stats(&self) -> Vec<&ModelStats> {
        let mut stats: Vec<_> = self
            .aggregates
            .values()
            .flat_map(|by_cat| by_cat.values())
            .collect();
        stats.sort_by(|a, b| a.model_name.cmp(&b.model_name));
        stats
    }

    fn recent_inferences(&self, limit: usize) -> Vec<&InferenceRecord> {
        self.inferences.iter().rev().take(limit).collect()
    }

    fn lifecycle_history(&self, model_name: &str) -> Vec<&LifecycleRecord> {
        self.lifecycles
            .iter()
            .filter(|r| r.model_name == model_name)
            .collect()
    }

    fn clear(&mut self) {
        self.inferences.clear();
        self.lifecycles.clear();
        self.aggregates.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::statistics::metrics::{InferenceDetail, LifecycleEvent};
    use std::time::SystemTime;

    fn text_record(name: &str, tokens: usize, time_ms: u64) -> InferenceRecord {
        InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: name.to_string(),
            category: ModelCategory::Chat,
            inference_time_ms: time_ms,
            detail: InferenceDetail::TextGeneration {
                tokens_generated: tokens,
                prompt_tokens: None,
                prefix_reuse_tokens: None,
            },
        }
    }

    #[test]
    fn record_and_aggregate() {
        let mut backend = InMemoryBackend::new(100);
        backend.record_inference(text_record("m1", 50, 250));
        backend.record_inference(text_record("m1", 100, 500));

        let stats = backend.model_stats("m1", ModelCategory::Chat).unwrap();
        assert_eq!(stats.inference_count, 2);
        assert_eq!(stats.total_inference_time_ms, 750);
        // Both should be 200 tok/s
        assert!((stats.primary_metric.mean() - 200.0).abs() < 0.01);
    }

    #[test]
    fn ring_buffer_eviction() {
        let mut backend = InMemoryBackend::new(3);
        for i in 0..5 {
            backend.record_inference(text_record("m", i * 10, 100));
        }
        assert_eq!(backend.inferences.len(), 3);
        let recent = backend.recent_inferences(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn all_stats_sorted() {
        let mut backend = InMemoryBackend::new(100);
        backend.record_inference(text_record("zebra", 10, 100));
        backend.record_inference(text_record("alpha", 10, 100));

        let all = backend.all_stats();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].model_name, "alpha");
        assert_eq!(all[1].model_name, "zebra");
    }

    #[test]
    fn lifecycle_tracking() {
        let mut backend = InMemoryBackend::new(100);
        backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: "m1".to_string(),
            event: LifecycleEvent::Loaded {
                load_time_ms: 1500,
                memory_bytes: 4_000_000_000,
            },
        });
        backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: "m2".to_string(),
            event: LifecycleEvent::Loaded {
                load_time_ms: 500,
                memory_bytes: 1_000_000_000,
            },
        });
        backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: "m1".to_string(),
            event: LifecycleEvent::Unloaded,
        });

        assert_eq!(backend.lifecycle_history("m1").len(), 2);
        assert_eq!(backend.lifecycle_history("m2").len(), 1);
        assert_eq!(backend.lifecycle_history("unknown").len(), 0);
    }

    #[test]
    fn clear_resets_all() {
        let mut backend = InMemoryBackend::new(100);
        backend.record_inference(text_record("m", 10, 100));
        backend.record_lifecycle(LifecycleRecord {
            timestamp: SystemTime::now(),
            model_name: "m".to_string(),
            event: LifecycleEvent::Unloaded,
        });

        backend.clear();
        assert!(backend.all_stats().is_empty());
        assert!(backend.recent_inferences(100).is_empty());
        assert!(backend.lifecycle_history("m").is_empty());
    }

    #[test]
    fn image_generation_secondary_metric() {
        let mut backend = InMemoryBackend::new(100);
        backend.record_inference(InferenceRecord {
            timestamp: SystemTime::now(),
            model_name: "flux".to_string(),
            category: ModelCategory::Imagine,
            inference_time_ms: 4000,
            detail: InferenceDetail::ImageGeneration {
                images_generated: 1,
                steps: 20,
                total_pixels: 512 * 512,
            },
        });

        let stats = backend.model_stats("flux", ModelCategory::Imagine).unwrap();
        assert!(stats.secondary_metric.is_some());
        let sec = stats.secondary_metric.as_ref().unwrap();
        // 20 steps * 1 image / 4s = 5 step/s
        assert!((sec.mean() - 5.0).abs() < 0.01);
    }
}
