use anyhow::Result;
use candle_core::Tensor;

/// Decode audio codes from a DAC-style audio encoder into PCM f32 samples.
///
/// The input `codes` tensor typically has shape `[1, n_codebooks, seq_len]`.
/// The output is a flat `Vec<f32>` of mono PCM samples.
pub fn decode_to_pcm(pcm_tensor: &Tensor) -> Result<Vec<f32>> {
    let samples: Vec<f32> = pcm_tensor.squeeze(0)?.squeeze(0)?.to_vec1()?;
    Ok(samples)
}
