//! Flow-matching Euler discrete scheduler for Qwen-Image-2512.

use local_inference_helpers::candle_core::{Result, Tensor};

pub(crate) const BASE_IMAGE_SEQ_LEN: usize = 256;
pub(crate) const MAX_IMAGE_SEQ_LEN: usize = 8192;
pub(crate) const BASE_SHIFT: f64 = 0.5;
pub(crate) const MAX_SHIFT: f64 = 0.9;
const SHIFT_TERMINAL: f64 = 0.02;
pub(crate) const NUM_TRAIN_TIMESTEPS: usize = 1000;

pub(crate) fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

fn time_shift(mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    if sigma >= 1.0 {
        return 1.0;
    }
    let e_mu = mu.exp();
    e_mu / (e_mu + (1.0 / sigma - 1.0))
}

#[derive(Debug, Clone)]
pub(crate) struct QwenImageScheduler {
    pub sigmas: Vec<f64>,
    step_index: usize,
}

impl QwenImageScheduler {
    pub fn new(num_inference_steps: usize, mu: f64) -> Self {
        let mut sigmas: Vec<f64> = (0..num_inference_steps)
            .map(|i| {
                let t = i as f64 / num_inference_steps as f64;
                1.0 * (1.0 - t) + SHIFT_TERMINAL * t
            })
            .collect();

        sigmas = sigmas.iter().map(|&s| time_shift(mu, s)).collect();
        sigmas.push(0.0);

        Self {
            sigmas,
            step_index: 0,
        }
    }

    pub fn current_timestep(&self) -> f64 {
        self.sigmas[self.step_index] * NUM_TRAIN_TIMESTEPS as f64
    }

    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;
        let prev_sample = (sample + (model_output * dt)?)?;
        self.step_index += 1;
        Ok(prev_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_shift_at_base_seq_len() {
        let mu = calculate_shift(
            BASE_IMAGE_SEQ_LEN,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!((mu - BASE_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn calculate_shift_at_max_seq_len() {
        let mu = calculate_shift(
            MAX_IMAGE_SEQ_LEN,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        assert!((mu - MAX_SHIFT).abs() < 1e-10);
    }

    #[test]
    fn time_shift_boundaries() {
        assert_eq!(time_shift(0.7, 0.0), 0.0);
        assert_eq!(time_shift(0.7, 1.0), 1.0);
    }

    #[test]
    fn scheduler_creates_correct_sigmas() {
        let scheduler = QwenImageScheduler::new(20, 0.7);
        assert_eq!(scheduler.sigmas.len(), 21);
        assert!(scheduler.sigmas[0] > 0.5);
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        for w in scheduler.sigmas.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
}
