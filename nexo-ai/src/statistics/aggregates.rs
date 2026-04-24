use crate::api::types::ModelCategory;
use serde::{Deserialize, Serialize};

/// Running statistics using Welford's online algorithm.
/// Tracks count, min, max, mean, and variance for a single scalar metric
/// without storing all individual values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningStat {
    pub count: u64,
    pub min: f64,
    pub max: f64,
    mean: f64,
    m2: f64,
}

impl RunningStat {
    pub fn new() -> Self {
        Self {
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.count += 1;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        // Welford's online algorithm
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for RunningStat {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated statistics for a specific (model, category) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub model_name: String,
    pub category: ModelCategory,
    pub inference_count: u64,
    pub total_inference_time_ms: u64,
    pub primary_metric: RunningStat,
    pub primary_metric_label: String,
    pub secondary_metric: Option<RunningStat>,
    pub secondary_metric_label: Option<String>,
}

impl ModelStats {
    pub fn new(model_name: String, category: ModelCategory, primary_label: &str) -> Self {
        Self {
            model_name,
            category,
            inference_count: 0,
            total_inference_time_ms: 0,
            primary_metric: RunningStat::new(),
            primary_metric_label: primary_label.to_string(),
            secondary_metric: None,
            secondary_metric_label: None,
        }
    }

    pub fn avg_inference_time_ms(&self) -> f64 {
        if self.inference_count == 0 {
            return 0.0;
        }
        self.total_inference_time_ms as f64 / self.inference_count as f64
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn running_stat_single_value() {
        let mut stat = RunningStat::new();
        stat.push(42.0);
        assert_eq!(stat.count, 1);
        assert!((stat.mean() - 42.0).abs() < f64::EPSILON);
        assert_eq!(stat.min, 42.0);
        assert_eq!(stat.max, 42.0);
        assert_eq!(stat.variance(), 0.0);
        assert_eq!(stat.std_dev(), 0.0);
    }

    #[test]
    fn running_stat_multiple_values() {
        let mut stat = RunningStat::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stat.push(v);
        }
        assert_eq!(stat.count, 8);
        assert!((stat.mean() - 5.0).abs() < 0.001);
        assert_eq!(stat.min, 2.0);
        assert_eq!(stat.max, 9.0);
        // Sample variance of [2,4,4,4,5,5,7,9] = 4.571...
        assert!((stat.variance() - 4.571).abs() < 0.01);
        assert!((stat.std_dev() - 2.138).abs() < 0.01);
    }

    #[test]
    fn running_stat_default() {
        let stat = RunningStat::default();
        assert_eq!(stat.count, 0);
        assert_eq!(stat.mean(), 0.0);
    }

    #[test]
    fn model_stats_avg_time() {
        let mut stats = ModelStats::new("test".to_string(), ModelCategory::Chat, "tok/s");
        stats.inference_count = 4;
        stats.total_inference_time_ms = 2000;
        assert!((stats.avg_inference_time_ms() - 500.0).abs() < 0.001);
    }

    #[test]
    fn model_stats_avg_time_zero() {
        let stats = ModelStats::new("test".to_string(), ModelCategory::Chat, "tok/s");
        assert_eq!(stats.avg_inference_time_ms(), 0.0);
    }
}
