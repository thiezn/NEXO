//! Qwen3-VL vision encoder loaded from mmproj GGUF.
//!
//! Architecture: ViT with fused QKV attention, GELU MLP, LayerNorm (with bias),
//! deepstack skip connections at specific layers, and a 2x2 spatial merger.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::with_tracing::QMatMul;

use crate::shared::types::*;
use crate::vision;

use super::pipeline::LoadedState;

// ── Constants ────────────────────────────────────────────────────────────────

const PATCH_SIZE: u32 = 16;
const MERGE_SIZE: u32 = 2;

// ── Layer norm with bias ────────────────────────────────────────────────────

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(x, &self.weight, &self.bias, self.eps)?)
    }
}

// ── ViT layer ────────────────────────────────────────────────────────────────

struct VitAttention {
    qkv: QMatMul,
    qkv_bias: Tensor,
    out: QMatMul,
    out_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl VitAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.broadcast_add(&self.qkv_bias)?;

        let qkv = qkv.reshape((b, seq, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let out = ctx.transpose(1, 2)?.contiguous()?.reshape((b, seq, ()))?;
        let out = self.out.forward(&out)?;
        let out = out.broadcast_add(&self.out_bias)?;
        Ok(out)
    }
}

struct VitMlp {
    up: QMatMul,
    up_bias: Tensor,
    down: QMatMul,
    down_bias: Tensor,
}

impl VitMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(x)?;
        let h = h.broadcast_add(&self.up_bias)?;
        let h = h.gelu_erf()?;
        let h = self.down.forward(&h)?;
        let h = h.broadcast_add(&self.down_bias)?;
        Ok(h)
    }
}

struct VitLayer {
    attn: VitAttention,
    mlp: VitMlp,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl VitLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + h)?;
        let h = self.ln2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((x + h)?)
    }
}

// ── Deepstack layer ─────────────────────────────────────────────────────────

struct DeepstackLayer {
    norm: LayerNorm,
    fc1: QMatMul,
    fc1_bias: Tensor,
    fc2: QMatMul,
    fc2_bias: Tensor,
}

impl DeepstackLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.broadcast_add(&self.fc1_bias)?;
        let h = h.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        let h = h.broadcast_add(&self.fc2_bias)?;
        Ok(h)
    }
}

// ── Merger ───────────────────────────────────────────────────────────────────

struct Merger {
    linear1: QMatMul,
    linear1_bias: Tensor,
    linear2: QMatMul,
    linear2_bias: Tensor,
}

impl Merger {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = h.broadcast_add(&self.linear1_bias)?;
        let h = h.gelu_erf()?;
        let h = self.linear2.forward(&h)?;
        let h = h.broadcast_add(&self.linear2_bias)?;
        Ok(h)
    }
}

// ── ViT model ────────────────────────────────────────────────────────────────

struct VisionModel {
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    position_embed: Tensor,
    layers: Vec<VitLayer>,
    post_ln: LayerNorm,
    deepstack_layers: Vec<(usize, DeepstackLayer)>,
    merger: Merger,
    merge_size: usize,
}

impl VisionModel {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Patch embedding via conv2d
        let patches = pixel_values.conv2d(
            &self.patch_embed_weight,
            0,                      // padding
            PATCH_SIZE as usize,    // stride
            1,                      // dilation
            1,                      // groups
        )?;
        let patches = patches.broadcast_add(
            &self.patch_embed_bias.reshape((1, (), 1, 1))?,
        )?;

        let (b, hidden, ph, pw) = patches.dims4()?;
        let seq = ph * pw;
        let mut x = patches.reshape((b, hidden, seq))?.transpose(1, 2)?;

        // Add position embeddings (truncate/pad if needed)
        let pos_len = self.position_embed.dim(1)?;
        if seq <= pos_len {
            let pos = self.position_embed.i((.., ..seq, ..))?;
            x = (x + pos)?;
        }

        // Run ViT blocks, collecting deepstack outputs
        let mut deepstack_outputs: Vec<Tensor> = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            for (ds_idx, ds_layer) in &self.deepstack_layers {
                if *ds_idx == layer_idx {
                    // Merge 2x2 patches before deepstack
                    let merged_x = self.spatial_merge(&x, ph, pw)?;
                    let ds_out = ds_layer.forward(&merged_x)?;
                    deepstack_outputs.push(ds_out);
                }
            }
        }

        // Post layer norm
        x = self.post_ln.forward(&x)?;

        // Merge 2x2 spatial patches
        let x = self.spatial_merge(&x, ph, pw)?;

        // Run merger (projection to text hidden dim)
        let x = self.merger.forward(&x)?;

        // Add deepstack contributions
        let mut result = x;
        for ds_out in &deepstack_outputs {
            result = (result + ds_out)?;
        }

        Ok(result)
    }

    fn spatial_merge(&self, x: &Tensor, ph: usize, pw: usize) -> Result<Tensor> {
        let (b, _seq, hidden) = x.dims3()?;
        let merge = self.merge_size;

        if ph >= merge && pw >= merge && ph % merge == 0 && pw % merge == 0 {
            let merged_ph = ph / merge;
            let merged_pw = pw / merge;
            let merged_seq = merged_ph * merged_pw;
            let merged_hidden = hidden * merge * merge;

            let reshaped = x.reshape((b, merged_ph, merge, merged_pw, merge, hidden))?;
            let permuted = reshaped.permute((0, 1, 3, 2, 4, 5))?;
            let merged = permuted.reshape((b, merged_seq, merged_hidden))?;
            Ok(merged)
        } else {
            Ok(x.clone())
        }
    }
}

// ── Vision state ─────────────────────────────────────────────────────────────

pub struct VisionState {
    vision_model: VisionModel,
    image_token_id: u32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

// ── GGUF helpers ─────────────────────────────────────────────────────────────

fn load_f32_tensor(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    let qt = content
        .tensor(file, name, device)
        .map_err(|e| anyhow::anyhow!("cannot find tensor: {name}: {e}"))?;
    qt.dequantize(device)?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

fn load_qmatmul(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<QMatMul> {
    let qt = content
        .tensor(file, name, device)
        .map_err(|e| anyhow::anyhow!("cannot find tensor: {name}: {e}"))?;
    QMatMul::from_weights(qt.into()).map_err(Into::into)
}

fn load_layer_norm(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    weight_name: &str,
    bias_name: &str,
    eps: f32,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = load_f32_tensor(content, file, weight_name, device)?;
    let bias = load_f32_tensor(content, file, bias_name, device)?;
    Ok(LayerNorm { weight, bias, eps })
}

// ── Load vision components from mmproj GGUF ──────────────────────────────────

pub fn load_vision(mmproj_path: &Path, device: &Device) -> Result<VisionState> {
    let (content, mut file) = crate::models::shared::weights::load_gguf(mmproj_path)?;

    let md = &content.metadata;
    let hidden_size = get_u32(md, "clip.vision.embedding_length")? as usize;
    let num_heads = get_u32(md, "clip.vision.attention.head_count")? as usize;
    let num_layers = get_u32(md, "clip.vision.block_count")? as usize;
    let head_dim = hidden_size / num_heads;
    let ln_eps = get_f32_or(md, "clip.vision.attention.layer_norm_epsilon", 1e-6);
    let image_token_id = get_u32_or(md, "clip.vision.image_token_id", 151655);
    let merge_size = get_u32_or(md, "clip.vision.spatial_merge_size", 2) as usize;

    let image_mean = get_f32_array_or(md, "clip.vision.image_mean", [0.5, 0.5, 0.5]);
    let image_std = get_f32_array_or(md, "clip.vision.image_std", [0.5, 0.5, 0.5]);

    // Deepstack layer indices
    let deepstack_flags: Vec<bool> = md
        .get("clip.vision.is_deepstack_layers")
        .and_then(|v| match v {
            gguf_file::Value::Array(arr) => {
                let bools: Vec<bool> = arr
                    .iter()
                    .filter_map(|v| match v {
                        gguf_file::Value::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .collect();
                if bools.len() == arr.len() {
                    Some(bools)
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap_or_default();

    // Patch embedding — GGUF dims are reversed, so already [out_channels, in_channels, kH, kW]
    let patch_embed_weight = load_f32_tensor(&content, &mut file, "v.patch_embd.weight", device)?;
    let patch_embed_bias = load_f32_tensor(&content, &mut file, "v.patch_embd.bias", device)?;

    // Position embeddings: GGUF dims [hidden, max_positions] = row-major [max_positions, hidden]
    let position_embed = load_f32_tensor(&content, &mut file, "v.position_embd.weight", device)?;
    let position_embed = position_embed.unsqueeze(0)?;

    // ViT layers
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("v.blk.{i}");
        let attn = VitAttention {
            qkv: load_qmatmul(&content, &mut file, &format!("{prefix}.attn_qkv.weight"), device)?,
            qkv_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.attn_qkv.bias"), device)?,
            out: load_qmatmul(&content, &mut file, &format!("{prefix}.attn_out.weight"), device)?,
            out_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.attn_out.bias"), device)?,
            num_heads,
            head_dim,
        };
        let mlp = VitMlp {
            up: load_qmatmul(&content, &mut file, &format!("{prefix}.ffn_up.weight"), device)?,
            up_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.ffn_up.bias"), device)?,
            down: load_qmatmul(&content, &mut file, &format!("{prefix}.ffn_down.weight"), device)?,
            down_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.ffn_down.bias"), device)?,
        };
        let ln1 = load_layer_norm(
            &content, &mut file,
            &format!("{prefix}.ln1.weight"),
            &format!("{prefix}.ln1.bias"),
            ln_eps, device,
        )?;
        let ln2 = load_layer_norm(
            &content, &mut file,
            &format!("{prefix}.ln2.weight"),
            &format!("{prefix}.ln2.bias"),
            ln_eps, device,
        )?;
        layers.push(VitLayer { attn, mlp, ln1, ln2 });
    }

    // Post layer norm
    let post_ln = load_layer_norm(
        &content, &mut file,
        "v.post_ln.weight", "v.post_ln.bias",
        ln_eps, device,
    )?;

    // Deepstack layers
    let mut deepstack_layers = Vec::new();
    for (i, &is_ds) in deepstack_flags.iter().enumerate() {
        if is_ds {
            let prefix = format!("v.deepstack.{i}");
            let ds = DeepstackLayer {
                norm: load_layer_norm(
                    &content, &mut file,
                    &format!("{prefix}.norm.weight"),
                    &format!("{prefix}.norm.bias"),
                    ln_eps, device,
                )?,
                fc1: load_qmatmul(&content, &mut file, &format!("{prefix}.fc1.weight"), device)?,
                fc1_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.fc1.bias"), device)?,
                fc2: load_qmatmul(&content, &mut file, &format!("{prefix}.fc2.weight"), device)?,
                fc2_bias: load_f32_tensor(&content, &mut file, &format!("{prefix}.fc2.bias"), device)?,
            };
            deepstack_layers.push((i, ds));
        }
    }

    // Merger (spatial merge + projection)
    let merger = Merger {
        linear1: load_qmatmul(&content, &mut file, "mm.0.weight", device)?,
        linear1_bias: load_f32_tensor(&content, &mut file, "mm.0.bias", device)?,
        linear2: load_qmatmul(&content, &mut file, "mm.2.weight", device)?,
        linear2_bias: load_f32_tensor(&content, &mut file, "mm.2.bias", device)?,
    };

    tracing::info!(
        "vision model loaded: {} layers, hidden_size={}, num_heads={}, deepstack_layers={}",
        num_layers,
        hidden_size,
        num_heads,
        deepstack_layers.len(),
    );

    Ok(VisionState {
        vision_model: VisionModel {
            patch_embed_weight,
            patch_embed_bias,
            position_embed,
            layers,
            post_ln,
            deepstack_layers,
            merger,
            merge_size,
        },
        image_token_id,
        image_mean,
        image_std,
    })
}

fn get_u32(md: &std::collections::HashMap<String, gguf_file::Value>, key: &str) -> Result<u32> {
    md.get(key)
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {key}"))
        .and_then(|v| v.to_u32().map_err(|e| anyhow::anyhow!("{e}")))
}

fn get_u32_or(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
    default: u32,
) -> u32 {
    md.get(key)
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(default)
}

fn get_f32_or(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
    default: f32,
) -> f32 {
    md.get(key)
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(default)
}

fn get_f32_array_or(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
    default: [f32; 3],
) -> [f32; 3] {
    md.get(key)
        .and_then(|v| match v {
            gguf_file::Value::Array(arr) if arr.len() >= 3 => {
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| match v {
                        gguf_file::Value::F32(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                if floats.len() >= 3 {
                    Some([floats[0], floats[1], floats[2]])
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap_or(default)
}

// ── Image preprocessing ──────────────────────────────────────────────────────

fn preprocess_image(
    image_data: &[u8],
    device: &Device,
    image_mean: &[f32; 3],
    image_std: &[f32; 3],
) -> Result<(Tensor, u32, u32)> {
    let img = image::load_from_memory(image_data).context("failed to decode image")?;
    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    let grid_unit = PATCH_SIZE * MERGE_SIZE; // 32
    let min_pixels = 256 * 28 * 28;
    let max_pixels = 1280 * 28 * 28;

    let buf = vision::ImageBuffer::from_rgb(rgb.into_raw(), orig_w, orig_h)?;
    let resized = vision::smart_resize(&buf, grid_unit, min_pixels, max_pixels)?;
    let (h, w) = (resized.height, resized.width);

    let norm_config = vision::NormalizeConfig {
        mean: *image_mean,
        std: *image_std,
    };
    let normalized = vision::normalize_rgb_f32(&resized, &norm_config)?;

    let tensor = Tensor::from_vec(normalized, (3, h as usize, w as usize), device)?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;

    Ok((tensor, h, w))
}

// ── Token injection ──────────────────────────────────────────────────────────

fn build_vision_tokens(
    tokenizer: &tokenizers::Tokenizer,
    user_text: &str,
    num_vision_tokens: usize,
    vision: &VisionState,
) -> Result<(Vec<u32>, usize, usize)> {
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("encode '{s}' failed: {e}"))?
            .get_ids()
            .to_vec())
    };

    let im_start = encode("<|im_start|>")?;
    let im_end = encode("<|im_end|>")?;
    let vision_start = encode("<|vision_start|>")?;
    let vision_end = encode("<|vision_end|>")?;
    let newline = encode("\n")?;

    let text_ids = tokenizer
        .encode(user_text, false)
        .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
        .get_ids()
        .to_vec();

    let mut tokens = Vec::new();

    // <|im_start|>user\n
    tokens.extend_from_slice(&im_start);
    tokens.extend(encode("user")?);
    tokens.extend_from_slice(&newline);

    // <|vision_start|>[image_pad x N]<|vision_end|>\n
    tokens.extend_from_slice(&vision_start);
    let img_start = tokens.len();
    tokens.extend(std::iter::repeat_n(vision.image_token_id, num_vision_tokens));
    let img_end = tokens.len();
    tokens.extend_from_slice(&vision_end);
    tokens.extend_from_slice(&newline);

    // {text}<|im_end|>\n
    tokens.extend_from_slice(&text_ids);
    tokens.extend_from_slice(&im_end);
    tokens.extend_from_slice(&newline);

    // <|im_start|>assistant\n
    tokens.extend_from_slice(&im_start);
    tokens.extend(encode("assistant")?);
    tokens.extend_from_slice(&newline);

    Ok((tokens, img_start, img_end))
}

fn replace_image_tokens(
    text_embeds: &Tensor,
    image_embeds: &Tensor,
    img_start: usize,
    img_end: usize,
) -> Result<Tensor> {
    let (_b, seq_len, _hidden) = text_embeds.dims3()?;
    let num_image_tokens = image_embeds.dim(1)?;
    let expected = img_end - img_start;

    if expected != num_image_tokens {
        anyhow::bail!(
            "image token count mismatch: {} positions vs {} image embeddings",
            expected,
            num_image_tokens,
        );
    }

    let mut parts: Vec<Tensor> = Vec::new();
    if img_start > 0 {
        parts.push(text_embeds.i((0..1, ..img_start, ..))?);
    }
    parts.push(image_embeds.clone());
    if img_end < seq_len {
        parts.push(text_embeds.i((0..1, img_end..seq_len, ..))?);
    }

    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

// ── Image analysis ───────────────────────────────────────────────────────────

pub fn analyze_image(
    state: &mut LoadedState,
    vision: &VisionState,
    request: &ImageAnalysisRequest,
) -> Result<ImageAnalysisResponse> {
    let start = std::time::Instant::now();
    state.weights.clear_kv_cache();

    let (pixel_values, _h, _w) =
        preprocess_image(&request.image_data, &state.device, &vision.image_mean, &vision.image_std)?;
    let image_embeds = vision.vision_model.forward(&pixel_values)?;
    let num_vision_tokens = image_embeds.dim(1)?;

    tracing::debug!(
        num_vision_tokens = num_vision_tokens,
        "vision encoding complete"
    );

    let (token_ids, img_start, img_end) =
        build_vision_tokens(&state.tokenizer, &request.prompt, num_vision_tokens, vision)?;

    let input_tensor = Tensor::new(&token_ids[..], &state.device)?.unsqueeze(0)?;
    let text_embeds = state.weights.embed_tokens(&input_tensor)?;
    let merged_embeds = replace_image_tokens(&text_embeds, &image_embeds, img_start, img_end)?;

    let seq_len = merged_embeds.dim(1)?;
    let logits = state.weights.forward_embeds(&merged_embeds, 0)?;
    let logits = logits.squeeze(0)?;
    let mut sampler = super::pipeline::create_sampler(request.temperature, 0.9, 0);
    let mut next_token = sampler.sample(&logits)?;

    let mut generated = vec![next_token];

    if state.eos_token_ids.contains(&next_token) {
        let elapsed = start.elapsed().as_millis() as u64;
        return Ok(ImageAnalysisResponse {
            text: state
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?,
            tokens_generated: generated.len(),
            inference_time_ms: elapsed,
        });
    }

    for i in 0..request.max_tokens.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &state.device)?.unsqueeze(0)?;
        let offset = seq_len + i + 1;
        let logits = state.weights.forward(&input, offset)?;
        let logits = logits.squeeze(0)?;
        next_token = sampler.sample(&logits)?;

        if state.eos_token_ids.contains(&next_token) {
            break;
        }
        generated.push(next_token);
    }

    let raw_text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    tracing::debug!(
        tokens = generated.len(),
        raw_text = %raw_text,
        "qwen3-vl image analysis raw output"
    );

    let text = super::template::strip_thinking(&raw_text);

    let elapsed = start.elapsed().as_millis() as u64;
    Ok(ImageAnalysisResponse {
        text,
        tokens_generated: generated.len(),
        inference_time_ms: elapsed,
    })
}
