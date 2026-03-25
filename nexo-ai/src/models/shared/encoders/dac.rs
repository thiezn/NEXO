use anyhow::Result;
use candle_core::Tensor;

/// Decode audio codes from a DAC-style audio encoder into PCM f32 samples.
///
/// The input tensor from `decode_codes` has shape `[1, 1, num_samples]`.
/// We apply `tanh` (missing from candle-transformers' DAC decoder but present
/// in the original Descript Audio Codec) to constrain values to [-1, 1], then
/// flatten to a `Vec<f32>`.
pub fn decode_to_pcm(pcm_tensor: &Tensor) -> Result<Vec<f32>> {
    let pcm = pcm_tensor.tanh()?;
    let samples: Vec<f32> = pcm.squeeze(0)?.squeeze(0)?.to_vec1()?;
    Ok(samples)
}
