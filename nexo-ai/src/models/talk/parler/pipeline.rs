use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::parler_tts;

use crate::models::shared::encoders::{dac, t5};
use crate::models::shared::weights::find_safetensor_files;
use crate::shared::types::TalkRequest;
use crate::shared::types::TalkResponse;

use super::config::normalize_config;

pub struct LoadedState {
    pub model: parler_tts::Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: parler_tts::Config,
    pub device: Device,
}

/// Check safetensors headers to determine what tensor transformations are needed.
struct TensorFixups {
    /// Parler-TTS mini v1.1 stores `decoder.lm_heads.weight` as a single
    /// concatenated tensor. Candle-transformers expects per-codebook tensors.
    needs_lm_heads_split: bool,
    /// Parler-TTS mini v1.1 stores fused `weight` tensors for the DAC audio
    /// encoder. Candle-transformers expects `weight_g` / `weight_v` (weight
    /// normalization) pairs.
    needs_weight_norm_decompose: bool,
    /// Parler-TTS mini v1.1 uses a different DAC key naming convention:
    /// `audio_encoder.encoder.block.N.res_unitM.conv1.weight` instead of
    /// `audio_encoder.model.encoder.block.N.block.M.block.K.weight_g`.
    needs_dac_key_remap: bool,
}

fn detect_fixups(files: &[std::path::PathBuf]) -> TensorFixups {
    let combined_key = "decoder.lm_heads.weight";
    let per_codebook_key = "decoder.lm_heads.0.weight";

    let mut needs_lm_heads_split = false;
    let mut needs_weight_norm_decompose = false;
    let mut needs_dac_key_remap = false;

    for f in files {
        let Some(header) = read_safetensors_header(f) else {
            continue;
        };
        if header.get(combined_key).is_some() && header.get(per_codebook_key).is_none() {
            needs_lm_heads_split = true;
        }

        if let Some(obj) = header.as_object() {
            // Check for mini-format DAC keys (no "model." prefix under audio_encoder).
            if !needs_dac_key_remap {
                needs_dac_key_remap = obj.keys().any(|k| {
                    k.starts_with("audio_encoder.")
                        && !k.starts_with("audio_encoder.model.")
                        && (k.starts_with("audio_encoder.encoder.")
                            || k.starts_with("audio_encoder.decoder.")
                            || k.starts_with("audio_encoder.quantizer."))
                });
            }

            // Check if any audio_encoder weight tensor lacks a weight_g companion
            // (indicates fused weights that need decomposition for candle-transformers).
            if !needs_weight_norm_decompose {
                let has_fused = obj.keys().any(|k| {
                    k.starts_with("audio_encoder.model.")
                        && k.ends_with(".weight")
                        && !k.ends_with(".weight_g")
                        && !k.ends_with(".weight_v")
                        && !obj.contains_key(&format!("{}_g", k))
                });
                if has_fused {
                    needs_weight_norm_decompose = true;
                }
            }
        }
    }

    // Mini format always uses fused weights that need decomposition after remapping.
    if needs_dac_key_remap {
        needs_weight_norm_decompose = true;
    }

    TensorFixups {
        needs_lm_heads_split,
        needs_weight_norm_decompose,
        needs_dac_key_remap,
    }
}

/// Read only the JSON header from a safetensors file (first 8 bytes = header
/// length, then header_len bytes of JSON). Avoids reading the entire multi-GB
/// tensor payload.
fn read_safetensors_header(path: &std::path::Path) -> Option<serde_json::Value> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).ok()?;
    serde_json::from_slice(&header_buf).ok()
}

/// Build a VarBuilder from safetensor files, applying any needed tensor
/// transformations for candle-transformers compatibility.
fn load_var_builder(
    files: &[std::path::PathBuf],
    config: &parler_tts::Config,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let fixups = detect_fixups(files);

    if !fixups.needs_lm_heads_split
        && !fixups.needs_weight_norm_decompose
        && !fixups.needs_dac_key_remap
    {
        return unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device) }
            .map_err(Into::into);
    }

    // Load all tensors so we can patch them.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for file in files {
        tensors.extend(candle_core::safetensors::load(file, device)?);
    }

    // Key remapping must happen first so subsequent fixups see the expected key format.
    if fixups.needs_dac_key_remap {
        tracing::info!("remapping parler-mini DAC keys to candle-transformers format");
        remap_mini_dac_keys(&mut tensors);
    }

    if fixups.needs_lm_heads_split {
        tracing::info!("splitting combined lm_heads tensor for candle-transformers compatibility");
        let combined_key = "decoder.lm_heads.weight";
        if let Some(combined) = tensors.remove(combined_key) {
            let num_codebooks = config.decoder.num_codebooks;
            let chunks = combined.chunk(num_codebooks, 0)?;
            for (i, chunk) in chunks.into_iter().enumerate() {
                tensors.insert(format!("decoder.lm_heads.{i}.weight"), chunk);
            }
        }
    }

    if fixups.needs_weight_norm_decompose {
        tracing::info!("decomposing fused DAC weights into weight_g/weight_v pairs");
        decompose_fused_weights(&mut tensors)?;
    }

    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

/// Decompose fused `weight` tensors into `weight_g` / `weight_v` pairs for
/// layers that candle-transformers expects in weight-norm form.
///
/// Only processes 3-dimensional tensors (Conv1d shape `[out_c, in_c, k]`) under
/// `audio_encoder.model` that lack a corresponding `weight_g` key.
fn decompose_fused_weights(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    let keys_to_decompose: Vec<String> = tensors
        .keys()
        .filter(|k| k.starts_with("audio_encoder.model.") && k.ends_with(".weight"))
        .filter(|k| {
            let wg_key = format!("{}_g", k);
            !tensors.contains_key(&wg_key)
        })
        .filter(|k| tensors.get(*k).is_some_and(|t| t.dims().len() == 3))
        .cloned()
        .collect();

    for key in keys_to_decompose {
        let Some(weight) = tensors.remove(&key) else {
            continue;
        };
        let (weight_g, weight_v) = decompose_weight_norm(&weight)?;
        let base = key.strip_suffix(".weight").unwrap_or(&key);
        tensors.insert(format!("{base}.weight_g"), weight_g);
        tensors.insert(format!("{base}.weight_v"), weight_v);
    }

    Ok(())
}

/// Given a fused weight tensor of shape `[out_c, in_c, k]`, decompose it into
/// `weight_g` (shape `[out_c, 1, 1]`) and `weight_v` (same as input) such that
/// `weight_v * weight_g / ||weight_v|| == weight`.
pub(crate) fn decompose_weight_norm(weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let weight_v = weight.clone();
    let weight_g = weight.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    Ok((weight_g, weight_v))
}

// ── Parler-mini DAC key remapping ────────────────────────────────────────────
//
// Parler-mini v1.1 stores DAC audio encoder weights with a different naming
// convention than what candle-transformers expects (which matches parler-large).
//
// Mini format:  audio_encoder.encoder.block.{N}.res_unit{M}.conv1.weight
// Candle needs: audio_encoder.model.encoder.block.{N+1}.block.{M-1}.block.1.weight_g
//
// This module remaps ALL `audio_encoder.*` keys (that lack a `model.` prefix)
// to the candle-expected format. Weight decomposition (fused → weight_g/weight_v)
// is handled separately by `decompose_fused_weights`.

/// Remap parler-mini DAC tensor keys to candle-transformers format in-place.
fn remap_mini_dac_keys(tensors: &mut HashMap<String, Tensor>) {
    let keys: Vec<String> = tensors
        .keys()
        .filter(|k| k.starts_with("audio_encoder.") && !k.starts_with("audio_encoder.model."))
        .cloned()
        .collect();

    if keys.is_empty() {
        return;
    }

    // Find the max encoder/decoder block indices to compute snake/final-conv positions.
    let max_enc_block = keys
        .iter()
        .filter_map(|k| k.strip_prefix("audio_encoder.encoder.block."))
        .filter_map(|rest| rest.split('.').next().and_then(|s| s.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);

    let max_dec_block = keys
        .iter()
        .filter_map(|k| k.strip_prefix("audio_encoder.decoder.block."))
        .filter_map(|rest| rest.split('.').next().and_then(|s| s.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);

    tracing::info!(
        "remapping {} DAC keys (enc blocks 0..={max_enc_block}, dec blocks 0..={max_dec_block})",
        keys.len()
    );

    for old_key in keys {
        let tensor = tensors.remove(&old_key).unwrap();
        let new_key = remap_single_dac_key(&old_key, max_enc_block, max_dec_block);
        tracing::debug!("remap: {old_key} → {new_key}");
        tensors.insert(new_key, tensor);
    }
}

fn remap_single_dac_key(key: &str, max_enc_block: usize, max_dec_block: usize) -> String {
    let rest = key
        .strip_prefix("audio_encoder.")
        .expect("key must start with audio_encoder.");

    if let Some(enc) = rest.strip_prefix("encoder.") {
        return remap_encoder(enc, max_enc_block);
    }
    if let Some(dec) = rest.strip_prefix("decoder.") {
        return remap_decoder(dec, max_dec_block);
    }
    if let Some(q) = rest.strip_prefix("quantizer.") {
        // Quantizer: just add model. prefix (structural layout matches).
        return format!("audio_encoder.model.quantizer.{q}");
    }
    // Fallback
    format!("audio_encoder.model.{rest}")
}

/// Remap a key under `audio_encoder.encoder.` to the candle index-based format.
///
/// Candle encoder layout (sequential Vec):
///   [0] initial Conv1d
///   [1..=N] EncoderBlocks (one per stride)
///   [N+1] final Snake1d
///   [N+2] final Conv1d
///
/// Each EncoderBlock layout:
///   [0..R-1] ResidualUnits
///   [R]      Snake1d
///   [R+1]    downsample Conv1d
///
/// Each ResidualUnit layout:
///   [0] Snake1d, [1] Conv1d, [2] Snake1d, [3] Conv1d
fn remap_encoder(rest: &str, max_block: usize) -> String {
    let pfx = "audio_encoder.model.encoder.block";

    if let Some(leaf) = rest.strip_prefix("conv1.") {
        return format!("{pfx}.0.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("snake1.") {
        return format!("{pfx}.{}.{leaf}", max_block + 2);
    }
    if let Some(leaf) = rest.strip_prefix("conv2.") {
        return format!("{pfx}.{}.{leaf}", max_block + 3);
    }
    if let Some(block_rest) = rest.strip_prefix("block.") {
        if let Some((n_str, after_n)) = block_rest.split_once('.') {
            if let Ok(n) = n_str.parse::<usize>() {
                return remap_encoder_block(pfx, n + 1, after_n);
            }
        }
    }
    format!("audio_encoder.model.encoder.{rest}")
}

fn remap_encoder_block(pfx: &str, ci: usize, rest: &str) -> String {
    if let Some(ru) = rest.strip_prefix("res_unit") {
        if let Some((m_str, after_m)) = ru.split_once('.') {
            if let Ok(m) = m_str.parse::<usize>() {
                // res_units are 1-indexed in mini format
                return remap_residual_unit(&format!("{pfx}.{ci}.block.{}.block", m - 1), after_m);
            }
        }
    }
    if let Some(leaf) = rest.strip_prefix("snake1.") {
        return format!("{pfx}.{ci}.block.3.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("conv1.") {
        return format!("{pfx}.{ci}.block.4.{leaf}");
    }
    format!("{pfx}.{ci}.{rest}")
}

/// Remap a key under `audio_encoder.decoder.` to the candle index-based format.
///
/// Candle decoder layout (sequential `model.N`):
///   [0] initial Conv1d, [1..=N] DecoderBlocks, [N+1] Snake1d, [N+2] final Conv1d
///
/// Each DecoderBlock: [0] Snake1d, [1] ConvTranspose1d, [2..2+R-1] ResidualUnits
fn remap_decoder(rest: &str, max_block: usize) -> String {
    let pfx = "audio_encoder.model.decoder.model";

    if let Some(leaf) = rest.strip_prefix("conv1.") {
        return format!("{pfx}.0.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("snake1.") {
        return format!("{pfx}.{}.{leaf}", max_block + 2);
    }
    if let Some(leaf) = rest.strip_prefix("conv2.") {
        return format!("{pfx}.{}.{leaf}", max_block + 3);
    }
    if let Some(block_rest) = rest.strip_prefix("block.") {
        if let Some((n_str, after_n)) = block_rest.split_once('.') {
            if let Ok(n) = n_str.parse::<usize>() {
                return remap_decoder_block(pfx, n + 1, after_n);
            }
        }
    }
    format!("audio_encoder.model.decoder.{rest}")
}

fn remap_decoder_block(pfx: &str, ci: usize, rest: &str) -> String {
    if let Some(leaf) = rest.strip_prefix("snake1.") {
        return format!("{pfx}.{ci}.block.0.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("conv_t1.") {
        return format!("{pfx}.{ci}.block.1.{leaf}");
    }
    if let Some(ru) = rest.strip_prefix("res_unit") {
        if let Some((m_str, after_m)) = ru.split_once('.') {
            if let Ok(m) = m_str.parse::<usize>() {
                // decoder res_units start at block index 2
                return remap_residual_unit(&format!("{pfx}.{ci}.block.{}.block", m + 1), after_m);
            }
        }
    }
    format!("{pfx}.{ci}.{rest}")
}

/// Remap a residual unit component to candle's indexed format.
///
/// Layout: [0] Snake1d, [1] Conv1d, [2] Snake1d, [3] Conv1d
fn remap_residual_unit(block_pfx: &str, rest: &str) -> String {
    if let Some(leaf) = rest.strip_prefix("snake1.") {
        return format!("{block_pfx}.0.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("conv1.") {
        return format!("{block_pfx}.1.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("snake2.") {
        return format!("{block_pfx}.2.{leaf}");
    }
    if let Some(leaf) = rest.strip_prefix("conv2.") {
        return format!("{block_pfx}.3.{leaf}");
    }
    format!("{block_pfx}.{rest}")
}

/// Load a Parler-TTS model from the given directory.
///
/// Expects `config.json`, `tokenizer.json`, and one or more `.safetensors` files.
pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let start = Instant::now();

    // Force CPU — candle-transformers' parler_tts::Model::generate() creates
    // internal u32 tensors that trigger Metal device mismatch in index_select.
    let device = Device::Cpu;
    tracing::info!("using CPU for Parler (Metal has u32 index_select issues)");
    let dtype = crate::device::gpu_dtype(&device);
    tracing::info!("device ready in {:.1}s", start.elapsed().as_secs_f64());

    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", config_path.display()))?;
    let normalized = normalize_config(&config_str)?;
    let config: parler_tts::Config = serde_json::from_str(&normalized)?;
    tracing::info!("config loaded");

    let safetensor_files = find_safetensor_files(model_dir)?;
    let vb = load_var_builder(&safetensor_files, &config, dtype, &device)?;
    let model = parler_tts::Model::new(&config, vb)?;
    tracing::info!(
        "model loaded in {:.1}s ({} safetensor file(s))",
        start.elapsed().as_secs_f64(),
        safetensor_files.len()
    );

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    tracing::info!(
        "Parler-TTS fully loaded in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    Ok(LoadedState {
        model,
        tokenizer,
        config,
        device,
    })
}

/// Run text-to-speech synthesis on an already-loaded model.
pub fn synthesize(state: &mut LoadedState, request: &TalkRequest) -> Result<TalkResponse> {
    let start = Instant::now();

    let prompt_tokens = t5::encode_text(&state.tokenizer, &request.text, &state.device)?;
    let description_tokens =
        t5::encode_text(&state.tokenizer, &request.voice_description, &state.device)?;

    let temperature = if request.temperature <= 0.0 {
        None
    } else {
        Some(request.temperature)
    };
    let lp = LogitsProcessor::new(request.seed, temperature, None);
    let audio_tokens =
        state
            .model
            .generate(&prompt_tokens, &description_tokens, lp, request.max_tokens)?;

    // generate() returns [num_codebooks, seq_len]; decode_codes expects [batch, num_codebooks, seq_len]
    let audio_tokens = audio_tokens.unsqueeze(0)?;
    let pcm_tensor = state.model.audio_encoder.decode_codes(&audio_tokens)?;
    let pcm_samples = dac::decode_to_pcm(&pcm_tensor)?;

    let sample_rate = state.config.audio_encoder.sampling_rate;
    let inference_time_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        "generated {:.1}s of audio in {:.1}s (sample_rate={})",
        pcm_samples.len() as f64 / sample_rate as f64,
        inference_time_ms as f64 / 1000.0,
        sample_rate
    );

    Ok(TalkResponse {
        pcm_samples,
        sample_rate,
        inference_time_ms,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn decompose_weight_norm_roundtrip() {
        let device = Device::Cpu;
        // Synthetic Conv1d weight: [out_c=4, in_c=3, kernel=5]
        let data: Vec<f32> = (0..60).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let weight = Tensor::from_vec(data, (4, 3, 5), &device).unwrap();

        let (weight_g, weight_v) = decompose_weight_norm(&weight).unwrap();

        assert_eq!(weight_g.dims(), &[4, 1, 1]);
        assert_eq!(weight_v.dims(), &[4, 3, 5]);

        // Reconstruct: weight_v * weight_g / ||weight_v||
        let norm_v = weight_v.sqr().unwrap().sum_keepdim((1, 2)).unwrap().sqrt().unwrap();
        let reconstructed = weight_v
            .broadcast_mul(&weight_g)
            .unwrap()
            .broadcast_div(&norm_v)
            .unwrap();

        let original: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();
        let rebuilt: Vec<f32> = reconstructed.flatten_all().unwrap().to_vec1().unwrap();

        for (a, b) in original.iter().zip(rebuilt.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch: original={a}, reconstructed={b}"
            );
        }
    }

    #[test]
    fn decompose_fused_weights_processes_dac_keys() {
        let device = Device::Cpu;
        let weight = Tensor::ones((4, 3, 5), candle_core::DType::F32, &device).unwrap();

        let mut tensors = HashMap::new();
        // DAC conv weight — should be decomposed.
        tensors.insert(
            "audio_encoder.model.encoder.block.0.weight".to_string(),
            weight.clone(),
        );
        // A weight_g already present — this one should NOT be decomposed.
        tensors.insert(
            "audio_encoder.model.decoder.model.0.weight".to_string(),
            weight.clone(),
        );
        tensors.insert(
            "audio_encoder.model.decoder.model.0.weight_g".to_string(),
            Tensor::ones((4, 1, 1), candle_core::DType::F32, &device).unwrap(),
        );
        // Non-DAC key — should be left alone.
        tensors.insert("decoder.lm_heads.0.weight".to_string(), weight.clone());
        // 2D tensor (embedding) under audio_encoder — should NOT be decomposed.
        tensors.insert(
            "audio_encoder.model.quantizer.quantizers.0.codebook.weight".to_string(),
            Tensor::ones((1024, 8), candle_core::DType::F32, &device).unwrap(),
        );

        decompose_fused_weights(&mut tensors).unwrap();

        // Encoder block weight should be replaced by weight_g and weight_v.
        assert!(!tensors.contains_key("audio_encoder.model.encoder.block.0.weight"));
        assert!(tensors.contains_key("audio_encoder.model.encoder.block.0.weight_g"));
        assert!(tensors.contains_key("audio_encoder.model.encoder.block.0.weight_v"));

        // Decoder weight that already had weight_g should still have original weight.
        assert!(tensors.contains_key("audio_encoder.model.decoder.model.0.weight"));

        // Non-DAC key untouched.
        assert!(tensors.contains_key("decoder.lm_heads.0.weight"));

        // 2D embedding untouched.
        assert!(tensors.contains_key(
            "audio_encoder.model.quantizer.quantizers.0.codebook.weight"
        ));
    }

    #[test]
    fn remap_mini_encoder_keys() {
        let max_enc = 3;
        let max_dec = 3;

        // Initial conv
        assert_eq!(
            remap_single_dac_key("audio_encoder.encoder.conv1.weight", max_enc, max_dec),
            "audio_encoder.model.encoder.block.0.weight"
        );
        assert_eq!(
            remap_single_dac_key("audio_encoder.encoder.conv1.bias", max_enc, max_dec),
            "audio_encoder.model.encoder.block.0.bias"
        );

        // Encoder block res_unit
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.encoder.block.0.res_unit1.snake1.alpha",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.encoder.block.1.block.0.block.0.alpha"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.encoder.block.0.res_unit1.conv1.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.encoder.block.1.block.0.block.1.weight"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.encoder.block.2.res_unit3.conv2.bias",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.encoder.block.3.block.2.block.3.bias"
        );

        // Block snake + downsample conv
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.encoder.block.0.snake1.alpha",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.encoder.block.1.block.3.alpha"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.encoder.block.0.conv1.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.encoder.block.1.block.4.weight"
        );

        // Final snake + conv
        assert_eq!(
            remap_single_dac_key("audio_encoder.encoder.snake1.alpha", max_enc, max_dec),
            "audio_encoder.model.encoder.block.5.alpha"
        );
        assert_eq!(
            remap_single_dac_key("audio_encoder.encoder.conv2.weight", max_enc, max_dec),
            "audio_encoder.model.encoder.block.6.weight"
        );
    }

    #[test]
    fn remap_mini_decoder_keys() {
        let max_enc = 3;
        let max_dec = 3;

        // Initial conv
        assert_eq!(
            remap_single_dac_key("audio_encoder.decoder.conv1.weight", max_enc, max_dec),
            "audio_encoder.model.decoder.model.0.weight"
        );

        // Decoder block
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.decoder.block.0.snake1.alpha",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.decoder.model.1.block.0.alpha"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.decoder.block.0.conv_t1.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.decoder.model.1.block.1.weight"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.decoder.block.0.res_unit1.snake1.alpha",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.decoder.model.1.block.2.block.0.alpha"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.decoder.block.0.res_unit1.conv1.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.decoder.model.1.block.2.block.1.weight"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.decoder.block.3.res_unit3.conv2.bias",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.decoder.model.4.block.4.block.3.bias"
        );

        // Final snake + conv
        assert_eq!(
            remap_single_dac_key("audio_encoder.decoder.snake1.alpha", max_enc, max_dec),
            "audio_encoder.model.decoder.model.5.alpha"
        );
        assert_eq!(
            remap_single_dac_key("audio_encoder.decoder.conv2.weight", max_enc, max_dec),
            "audio_encoder.model.decoder.model.6.weight"
        );
    }

    #[test]
    fn remap_mini_quantizer_keys() {
        let max_enc = 3;
        let max_dec = 3;

        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.quantizer.quantizers.0.in_proj.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.quantizer.quantizers.0.in_proj.weight"
        );
        assert_eq!(
            remap_single_dac_key(
                "audio_encoder.quantizer.quantizers.0.codebook.weight",
                max_enc,
                max_dec
            ),
            "audio_encoder.model.quantizer.quantizers.0.codebook.weight"
        );
    }
}
