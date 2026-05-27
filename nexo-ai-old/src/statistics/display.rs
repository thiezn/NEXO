use super::aggregates::ModelStats;
use super::metrics::{LifecycleEvent, LifecycleRecord};

/// Print a summary table of all model statistics to stdout.
pub fn print_stats_table(stats: &[&ModelStats]) {
    if stats.is_empty() {
        println!("  no inference statistics recorded yet");
        return;
    }

    println!(
        "  {:<22} {:<10} {:>6} {:>10}  METRIC",
        "MODEL", "CATEGORY", "CALLS", "AVG TIME"
    );
    println!("  {}", "-".repeat(70));

    for s in stats {
        let avg_time = format_duration_ms(s.avg_inference_time_ms());
        let metric = format_metric(s.primary_metric.mean(), &s.primary_metric_label);

        let secondary = s
            .secondary_metric
            .as_ref()
            .zip(s.secondary_metric_label.as_ref())
            .map(|(stat, label)| format!(", {}", format_metric(stat.mean(), label)))
            .unwrap_or_default();

        println!(
            "  {:<22} {:<10} {:>6} {:>10}  {}{}",
            s.model_name, s.category, s.inference_count, avg_time, metric, secondary
        );
    }
}

/// Print detailed stats for a single model.
pub fn print_model_detail(stats: &[&ModelStats], lifecycle: &[&LifecycleRecord]) {
    if stats.is_empty() {
        println!("  no statistics for this model");
        return;
    }

    for s in stats {
        println!("  {} [{}]", s.model_name, s.category);
        println!("    calls:       {}", s.inference_count);
        println!(
            "    avg time:    {}",
            format_duration_ms(s.avg_inference_time_ms())
        );
        println!(
            "    {} avg: {:.2}  min: {:.2}  max: {:.2}  stddev: {:.2}",
            s.primary_metric_label,
            s.primary_metric.mean(),
            s.primary_metric.min,
            s.primary_metric.max,
            s.primary_metric.std_dev()
        );

        if let Some((sec, label)) = s
            .secondary_metric
            .as_ref()
            .zip(s.secondary_metric_label.as_ref())
        {
            println!(
                "    {} avg: {:.2}  min: {:.2}  max: {:.2}",
                label,
                sec.mean(),
                sec.min,
                sec.max
            );
        }
        println!();
    }

    if !lifecycle.is_empty() {
        println!("  lifecycle:");
        for event in lifecycle {
            match &event.event {
                LifecycleEvent::Loaded {
                    load_time_ms,
                    memory_bytes,
                } => {
                    println!(
                        "    loaded in {}  ({:.1} GB)",
                        format_duration_ms(*load_time_ms as f64),
                        *memory_bytes as f64 / 1_000_000_000.0
                    );
                }
                LifecycleEvent::Unloaded => {
                    println!("    unloaded");
                }
            }
        }
    }
}

/// Format a single inference result line for display after inference.
pub fn format_inference_result(inference_time_ms: u64, primary_value: f64, label: &str) -> String {
    format!(
        "  {} | {:.1} {}",
        format_duration_ms(inference_time_ms as f64),
        primary_value,
        label
    )
}

fn format_metric(value: f64, label: &str) -> String {
    format!("{:.1} {}", value, label)
}

fn format_duration_ms(ms: f64) -> String {
    if ms < 1000.0 {
        format!("{:.0}ms", ms)
    } else if ms < 60_000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        let mins = (ms / 60_000.0).floor();
        let secs = (ms - mins * 60_000.0) / 1000.0;
        format!("{:.0}m{:.0}s", mins, secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_milliseconds() {
        assert_eq!(format_duration_ms(42.0), "42ms");
        assert_eq!(format_duration_ms(999.0), "999ms");
    }

    #[test]
    fn format_duration_seconds() {
        assert_eq!(format_duration_ms(1500.0), "1.5s");
        assert_eq!(format_duration_ms(4200.0), "4.2s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration_ms(90_000.0), "1m30s");
    }

    #[test]
    fn format_inference_result_output() {
        let result = format_inference_result(320, 18.7, "tok/s");
        assert!(result.contains("320ms"));
        assert!(result.contains("18.7 tok/s"));
    }
}
