use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::AudioBuffer;

/// Configuration for microphone recording.
pub struct RecordConfig {
    /// Desired sample rate in Hz.
    pub sample_rate: u32,
    /// Maximum recording duration. `None` means unlimited (stop on silence or manually).
    pub max_duration_secs: Option<f64>,
    /// Stop after this many seconds of continuous silence. `None` disables silence detection.
    pub silence_threshold_secs: Option<f64>,
    /// RMS level below which audio is considered silence.
    pub silence_rms_threshold: f32,
}

impl Default for RecordConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            max_duration_secs: None,
            silence_threshold_secs: Some(2.0),
            silence_rms_threshold: 0.01,
        }
    }
}

/// Record audio from the default input device (microphone).
///
/// Blocks until the maximum duration is reached or silence is detected.
/// Returns a mono `AudioBuffer` at the requested sample rate (resampled if
/// the hardware rate differs).
pub fn record_microphone(config: &RecordConfig) -> anyhow::Result<AudioBuffer> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default audio input device found"))?;

    tracing::info!(device = ?device.description(), "using input device");

    let supported_config = device.default_input_config()?;
    let device_sample_rate = supported_config.sample_rate();
    let device_channels = supported_config.channels();

    let stream_config = cpal::StreamConfig {
        channels: device_channels,
        sample_rate: device_sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let samples_writer = Arc::clone(&samples);
    let stop_reader = Arc::clone(&stop_flag);
    let ch = device_channels as usize;
    let sr = device_sample_rate;
    let max_samples = config.max_duration_secs.map(|d| (d * sr as f64) as usize);
    let silence_frames = config
        .silence_threshold_secs
        .map(|d| (d * sr as f64) as usize);
    let rms_threshold = config.silence_rms_threshold;

    // Track consecutive silent frames for silence detection
    let silent_count = Arc::new(AtomicUsize::new(0));
    let silent_count_writer = Arc::clone(&silent_count);

    let err_flag = Arc::new(Mutex::new(None::<String>));
    let err_flag_cb = Arc::clone(&err_flag);

    let stream = device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if stop_reader.load(Ordering::Relaxed) {
                return;
            }

            // Down-mix to mono inline
            let mono: Vec<f32> = data
                .chunks_exact(ch)
                .map(|frame| {
                    let sum: f32 = frame.iter().sum();
                    sum / ch as f32
                })
                .collect();

            // Silence detection: compute RMS of this chunk
            if let Some(max_silent) = silence_frames {
                let rms = if mono.is_empty() {
                    0.0
                } else {
                    let sum_sq: f32 = mono.iter().map(|s| s * s).sum();
                    (sum_sq / mono.len() as f32).sqrt()
                };

                if rms < rms_threshold {
                    let prev = silent_count_writer.fetch_add(mono.len(), Ordering::Relaxed);
                    if prev + mono.len() >= max_silent {
                        stop_reader.store(true, Ordering::Relaxed);
                        return;
                    }
                } else {
                    silent_count_writer.store(0, Ordering::Relaxed);
                }
            }

            if let Ok(mut buf) = samples_writer.lock() {
                buf.extend_from_slice(&mono);

                // Check max duration
                if let Some(max) = max_samples
                    && buf.len() >= max
                {
                    stop_reader.store(true, Ordering::Relaxed);
                }
            }
        },
        move |err| {
            if let Ok(mut e) = err_flag_cb.lock() {
                *e = Some(format!("audio input stream error: {err}"));
            }
        },
        None,
    )?;

    stream.play()?;
    tracing::info!(
        sample_rate = device_sample_rate,
        channels = device_channels,
        "recording started"
    );

    // Poll until stop flag is set
    while !stop_flag.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);

    // Check for stream errors
    if let Ok(guard) = err_flag.lock()
        && let Some(ref msg) = *guard
    {
        anyhow::bail!("{msg}");
    }

    let recorded = match samples.lock() {
        Ok(guard) => guard.clone(),
        Err(e) => anyhow::bail!("failed to read recorded samples: {e}"),
    };

    // Truncate to max_samples if we overshot
    let recorded = if let Some(max) = max_samples {
        if recorded.len() > max {
            recorded[..max].to_vec()
        } else {
            recorded
        }
    } else {
        recorded
    };

    tracing::info!(
        frames = recorded.len(),
        device_sample_rate,
        "recording finished"
    );

    let buffer = AudioBuffer::new(recorded, device_sample_rate, 1);

    // Resample to the requested rate if the device rate differs
    if device_sample_rate != config.sample_rate {
        super::resample::resample(buffer, config.sample_rate)
    } else {
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_config_defaults() {
        let cfg = RecordConfig::default();
        assert_eq!(cfg.sample_rate, 16_000);
        assert!(cfg.max_duration_secs.is_none());
        assert!((cfg.silence_threshold_secs.unwrap_or(0.0) - 2.0).abs() < f64::EPSILON);
        assert!((cfg.silence_rms_threshold - 0.01).abs() < f32::EPSILON);
    }
}
