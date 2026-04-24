//! Load an mmproj GGUF file and build a `VarBuilder` with tensor names
//! matching the safetensors naming convention expected by `VisionTower`,
//! `AudioModel`, and `MultimodalEmbedder`.
//!
//! The mmproj GGUF uses a compact naming scheme (`v.blk.0.attn_q.weight`,
//! `a.blk.0.ffn_up.weight`, `mm.input_projection.weight`, etc.) while our
//! model constructors expect HuggingFace-style names (`vision_tower.encoder.
//! layers.0.self_attn.q_proj.weight`, etc.).  This module handles the
//! translation, dequantization, and any necessary reshaping.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use crate::models::support::weights::load_gguf;

/// Suffixes we skip — these are quantization calibration scalars, not model weights.
const SKIP_SUFFIXES: &[&str] = &["input_max", "input_min", "output_max", "output_min"];

/// Load an mmproj GGUF file, dequantize all tensors, map names to the
/// safetensors convention our model constructors expect, and return a
/// `VarBuilder` ready to be passed (via `.pp(...)`) to `VisionTower::new`,
/// `AudioModel::new`, and `MultimodalEmbedder::new`.
///
/// Load the mmproj GGUF and return the mapped tensors (in F32) plus
/// the device. Callers create VarBuilders at whatever dtype they need.
pub fn load_mmproj_tensors(mmproj_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let (content, mut file) = load_gguf(mmproj_path)?;

    let has_audio = content
        .metadata
        .get("clip.has_audio_encoder")
        .and_then(|v| v.to_bool().ok())
        .unwrap_or(false);

    tracing::info!(
        "mmproj GGUF: {} tensors, has_audio={}",
        content.tensor_infos.len(),
        has_audio,
    );

    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();

    for gguf_name in &names {
        if SKIP_SUFFIXES.iter().any(|s| gguf_name.ends_with(s)) {
            tracing::trace!("mmproj skip calibration tensor: {gguf_name}");
            continue;
        }

        if !gguf_name.ends_with(".weight") && !gguf_name.ends_with(".bias") {
            tracing::trace!("mmproj skip non-weight tensor: {gguf_name}");
            continue;
        }

        let mapped_name = match map_tensor_name(gguf_name) {
            Some(name) => name,
            None => {
                tracing::warn!("mmproj: unmapped tensor '{gguf_name}', skipping");
                continue;
            }
        };

        let qtensor = content
            .tensor(&mut file, gguf_name, device)
            .map_err(|e| anyhow::anyhow!("failed to load tensor '{gguf_name}': {e}"))?;

        let mut tensor = qtensor
            .dequantize(device)
            .map_err(|e| anyhow::anyhow!("failed to dequantize '{gguf_name}': {e}"))?;

        tensor = apply_reshape(gguf_name, tensor)
            .with_context(|| format!("reshape failed for '{gguf_name}'"))?;

        tracing::debug!(
            "mmproj: '{}' -> '{}' {:?}",
            gguf_name,
            mapped_name,
            tensor.shape(),
        );

        tensors.insert(mapped_name, tensor);
    }

    tracing::info!("mmproj: mapped {} tensors", tensors.len());
    Ok(tensors)
}

// ── Name mapping ────────────────────────────────────────────────────────────

/// Map a GGUF mmproj tensor name to the safetensors-compatible name our model
/// constructors expect.  Returns `None` for unknown patterns.
fn map_tensor_name(gguf_name: &str) -> Option<String> {
    // ── Vision block tensors (`v.blk.{i}.*`) ────────────────────────────
    if let Some(rest) = gguf_name.strip_prefix("v.blk.") {
        return map_vision_block(rest);
    }

    // ── Vision global tensors (`v.*`) ───────────────────────────────────
    if gguf_name == "v.patch_embd.weight" {
        return Some("vision_tower.patch_embedder.input_proj.weight".to_string());
    }
    if gguf_name == "v.position_embd.weight" {
        return Some("vision_tower.patch_embedder.position_embedding_table".to_string());
    }

    // ── Audio block tensors (`a.blk.{i}.*`) ─────────────────────────────
    if let Some(rest) = gguf_name.strip_prefix("a.blk.") {
        return map_audio_block(rest);
    }

    // ── Audio subsample conv projection (`a.conv1d.*`) ───────────────────
    if let Some(rest) = gguf_name.strip_prefix("a.conv1d.") {
        return map_audio_sscp(rest);
    }

    // ── Audio SSCP input projection ─────────────────────────────────────
    if gguf_name == "a.input_projection.weight" {
        return Some("audio_tower.subsample_conv_projection.input_proj_linear.weight".to_string());
    }

    // ── Audio output projection (`a.pre_encode.out.*`) ──────────────────
    if gguf_name == "a.pre_encode.out.weight" {
        return Some("audio_tower.output_proj.weight".to_string());
    }
    if gguf_name == "a.pre_encode.out.bias" {
        return Some("audio_tower.output_proj.bias".to_string());
    }

    // ── Vision multimodal projector (`mm.input_projection.*`) ───────────
    if gguf_name == "mm.input_projection.weight" {
        return Some("embed_vision.embedding_projection.weight".to_string());
    }

    // ── Audio multimodal projector (`mm.a.input_projection.*`) ──────────
    if gguf_name == "mm.a.input_projection.weight" {
        return Some("embed_audio.embedding_projection.weight".to_string());
    }

    None
}

/// Parse `"{i}.suffix"` from a string, returning `(layer_index, suffix)`.
fn parse_layer_suffix(s: &str) -> Option<(usize, &str)> {
    let dot = s.find('.')?;
    let idx: usize = s[..dot].parse().ok()?;
    let suffix = &s[dot + 1..];
    Some((idx, suffix))
}

/// Map a vision block tensor: input is everything after `"v.blk."`.
fn map_vision_block(rest: &str) -> Option<String> {
    let (i, suffix) = parse_layer_suffix(rest)?;
    let prefix = format!("vision_tower.encoder.layers.{i}");

    let mapped_suffix = match suffix {
        // Attention projections
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_out.weight" => "self_attn.o_proj.weight",
        // Q/K norms
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // MLP
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // Layer norms
        "ln1.weight" => "input_layernorm.weight",
        "attn_post_norm.weight" => "post_attention_layernorm.weight",
        "ln2.weight" => "pre_feedforward_layernorm.weight",
        "ffn_post_norm.weight" => "post_feedforward_layernorm.weight",
        _ => return None,
    };

    Some(format!("{prefix}.{mapped_suffix}"))
}

/// Map an audio block tensor: input is everything after `"a.blk."`.
fn map_audio_block(rest: &str) -> Option<String> {
    let (i, suffix) = parse_layer_suffix(rest)?;
    let prefix = format!("audio_tower.layers.{i}");

    let mapped_suffix = match suffix {
        // Attention projections
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_out.weight" => "self_attn.post.weight",
        // Relative position K
        "attn_k_rel.weight" => "self_attn.relative_k_proj.weight",
        // Per-dim scale (note: no ".weight" suffix in target)
        "per_dim_scale.weight" => "self_attn.per_dim_scale",
        // Attention norms
        "attn_pre_norm.weight" => "norm_pre_attn.weight",
        "attn_post_norm.weight" => "norm_post_attn.weight",
        // Feed-forward 1
        "ffn_up.weight" => "feed_forward1.ffw_layer_1.weight",
        "ffn_down.weight" => "feed_forward1.ffw_layer_2.weight",
        "ffn_norm.weight" => "feed_forward1.pre_layer_norm.weight",
        "ffn_post_norm.weight" => "feed_forward1.post_layer_norm.weight",
        // Feed-forward 2
        "ffn_up_1.weight" => "feed_forward2.ffw_layer_1.weight",
        "ffn_down_1.weight" => "feed_forward2.ffw_layer_2.weight",
        "ffn_norm_1.weight" => "feed_forward2.pre_layer_norm.weight",
        "ffn_post_norm_1.weight" => "feed_forward2.post_layer_norm.weight",
        // LightConv1d
        "conv_pw1.weight" => "lconv1d.linear_start.weight",
        "conv_pw2.weight" => "lconv1d.linear_end.weight",
        "conv_dw.weight" => "lconv1d.depthwise_conv1d.weight",
        "conv_norm.weight" => "lconv1d.conv_norm.weight",
        "norm_conv.weight" => "lconv1d.pre_layer_norm.weight",
        // Output norm
        "ln2.weight" => "norm_out.weight",
        _ => return None,
    };

    Some(format!("{prefix}.{mapped_suffix}"))
}

/// Map an audio SSCP tensor: input is everything after `"a.conv1d."`.
/// GGUF names: `a.conv1d.0.weight`, `a.conv1d.0.norm.weight`, etc.
/// Model names: `audio_tower.subsample_conv_projection.layer{i}.conv.weight`, etc.
fn map_audio_sscp(rest: &str) -> Option<String> {
    let mapped = match rest {
        "0.weight" => "audio_tower.subsample_conv_projection.layer0.conv.weight",
        "0.norm.weight" => "audio_tower.subsample_conv_projection.layer0.norm.weight",
        "1.weight" => "audio_tower.subsample_conv_projection.layer1.conv.weight",
        "1.norm.weight" => "audio_tower.subsample_conv_projection.layer1.norm.weight",
        _ => return None,
    };
    Some(mapped.to_string())
}

// ── Tensor reshaping ────────────────────────────────────────────────────────

/// Apply special reshapes for tensors that differ in layout between GGUF and
/// the HuggingFace safetensors convention.
fn apply_reshape(gguf_name: &str, tensor: Tensor) -> Result<Tensor> {
    match gguf_name {
        // Patch embedding: GGUF stores [768, 3, 16, 16] (out, channels, patch_h, patch_w).
        // Model expects [768, 768] = [out_features, in_features] where in = 3*16*16 = 768.
        // Just flatten the last 3 dims.
        "v.patch_embd.weight" => {
            let shape = tensor.dims().to_vec();
            tracing::debug!(
                "mmproj reshape v.patch_embd.weight: {:?} -> [{}, {}]",
                shape,
                shape[0],
                shape[1..].iter().product::<usize>(),
            );
            let out_features = shape[0];
            let in_features: usize = shape[1..].iter().product();
            tensor
                .reshape((out_features, in_features))
                .map_err(|e| anyhow::anyhow!("patch_embd reshape: {e}"))
        }

        // Position embedding table: GGUF stores [2, 10240, 768] which is already
        // the shape our model expects.  No reshape needed.
        "v.position_embd.weight" => {
            tracing::debug!(
                "mmproj v.position_embd.weight: {:?} (no reshape needed)",
                tensor.dims(),
            );
            Ok(tensor)
        }

        // Audio depthwise conv weights: GGUF stores [groups, kernel_size] but
        // candle Conv1d expects [groups, 1, kernel_size].
        name if name.starts_with("a.blk.") && name.ends_with(".conv_dw.weight") => {
            let shape = tensor.dims();
            if shape.len() == 2 {
                tracing::debug!(
                    "mmproj unsqueeze {}: {:?} -> [{}, 1, {}]",
                    gguf_name,
                    shape,
                    shape[0],
                    shape[1],
                );
                tensor
                    .unsqueeze(1)
                    .map_err(|e| anyhow::anyhow!("conv_dw unsqueeze: {e}"))
            } else {
                Ok(tensor)
            }
        }

        // Everything else: no reshape needed.
        _ => Ok(tensor),
    }
}
