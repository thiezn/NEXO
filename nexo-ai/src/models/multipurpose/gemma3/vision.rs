use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;

use crate::shared::types::*;

use super::pipeline::LoadedState;

// ── Vision config (extracted from HF config.json) ───────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3VisionHfConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
}

impl Gemma3VisionHfConfig {
    pub fn to_siglip_config(&self) -> siglip::VisionConfig {
        siglip::VisionConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_channels: 3,
            image_size: self.image_size,
            patch_size: self.patch_size,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }
}

// ── Loaded vision state ─────────────────────────────────────────────────────

/// Pre-resolved token IDs for structural tokens used in vision prompts.
pub struct StructuralTokenIds {
    pub start_turn: Vec<u32>,
    pub end_turn: Vec<u32>,
    pub user: Vec<u32>,
    pub model: Vec<u32>,
    pub newline: Vec<u32>,
}

pub struct VisionState {
    pub vision_model: siglip::VisionModel,
    pub projection_weight: Tensor,
    pub soft_emb_norm_weight: Tensor,
    pub vision_config: siglip::VisionConfig,
    pub mm_tokens_per_image: usize,
    pub image_token_index: u32,
    pub structural_tokens: StructuralTokenIds,
}

// ── Load vision components ──────────────────────────────────────────────────

pub fn load_vision(
    vb: &VarBuilder,
    config_str: &str,
    _device: &Device,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<VisionState> {
    #[derive(serde::Deserialize)]
    struct HfTextConfig {
        hidden_size: usize,
    }
    #[derive(serde::Deserialize)]
    struct HfRoot {
        vision_config: Gemma3VisionHfConfig,
        text_config: HfTextConfig,
        #[serde(default = "default_mm_tokens")]
        mm_tokens_per_image: usize,
        #[serde(default = "default_image_token_index")]
        image_token_index: u32,
    }
    fn default_mm_tokens() -> usize { 256 }
    fn default_image_token_index() -> u32 { 262144 }

    let hf_root: HfRoot =
        serde_json::from_str(config_str).context("failed to parse vision_config")?;
    let siglip_cfg = hf_root.vision_config.to_siglip_config();
    let text_hidden = hf_root.text_config.hidden_size;

    let vb_vision = vb.pp("vision_tower").pp("vision_model");
    let vision_model = siglip::VisionModel::new(&siglip_cfg, false, vb_vision)?;

    let vb_proj = vb.pp("multi_modal_projector");
    // Weight shape is [vision_hidden, text_hidden] — used as input @ weight.
    let projection_weight = vb_proj.get(
        (siglip_cfg.hidden_size, text_hidden),
        "mm_input_projection_weight",
    )?;
    let soft_emb_norm_weight = vb_proj
        .pp("mm_soft_emb_norm")
        .get(siglip_cfg.hidden_size, "weight")?;

    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("encode '{s}' failed: {e}"))?
            .get_ids()
            .to_vec())
    };
    let structural_tokens = StructuralTokenIds {
        start_turn: encode("<start_of_turn>")?,
        end_turn: encode("<end_of_turn>")?,
        user: encode("user")?,
        model: encode("model")?,
        newline: encode("\n")?,
    };

    tracing::info!(
        "vision encoder loaded: {} patches, pooled to {} tokens",
        siglip_cfg.num_patches(),
        hf_root.mm_tokens_per_image,
    );

    Ok(VisionState {
        vision_model,
        projection_weight,
        soft_emb_norm_weight,
        vision_config: siglip_cfg,
        mm_tokens_per_image: hf_root.mm_tokens_per_image,
        image_token_index: hf_root.image_token_index,
        structural_tokens,
    })
}

// ── Image preprocessing ─────────────────────────────────────────────────────

/// Convert raw image bytes to a normalized pixel tensor [1, 3, H, W].
/// Resizes to the model's expected image_size and normalizes with
/// SigLIP mean/std (0.5, 0.5).
pub fn preprocess_image(
    image_data: &[u8],
    image_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let img = image::load_from_memory(image_data)
        .context("failed to decode image")?
        .resize_exact(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();

    let pixels: Vec<f32> = img.pixels().flat_map(|p| {
        let [r, g, b] = p.0;
        [
            (r as f32 / 255.0 - 0.5) / 0.5,
            (g as f32 / 255.0 - 0.5) / 0.5,
            (b as f32 / 255.0 - 0.5) / 0.5,
        ]
    }).collect();

    // HWC → CHW
    let h = image_size;
    let w = image_size;
    let tensor = Tensor::from_vec(pixels, (h, w, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(DType::BF16)?;

    Ok(tensor)
}

// ── Vision encoding ─────────────────────────────────────────────────────────

/// Run vision encoder → average pool → project → L2 normalize.
/// Returns [1, mm_tokens_per_image, text_hidden_size].
pub fn encode_image(
    vision: &VisionState,
    pixel_values: &Tensor,
) -> Result<Tensor> {
    // SigLIP encoder: [1, num_patches, vision_hidden]
    let features = vision.vision_model.forward(pixel_values)?;

    // Average pool from num_patches to mm_tokens_per_image.
    // Patches form a grid: (image_size / patch_size)²
    let grid_size = vision.vision_config.image_size / vision.vision_config.patch_size;
    let target_grid = (vision.mm_tokens_per_image as f64).sqrt() as usize; // 16
    let pool_kernel = grid_size / target_grid; // 64/16 = 4

    // [1, 4096, 1152] → [1, 1152, 64, 64]
    let (b, _seq, hidden) = features.dims3()?;
    let features = features
        .transpose(1, 2)?
        .reshape((b, hidden, grid_size, grid_size))?;

    // Average pool 2D: kernel=4, stride=4 → [1, 1152, 16, 16]
    let pooled = features.avg_pool2d_with_stride(pool_kernel, pool_kernel)?;

    // [1, 1152, 16, 16] → [1, 256, 1152]
    let pooled = pooled
        .flatten_from(2)?
        .transpose(1, 2)?;

    // RMS norm (mm_soft_emb_norm)
    let normed = rms_norm(&pooled, &vision.soft_emb_norm_weight, 1e-6)?;

    // Linear projection: [1, 256, 1152] @ [1152, text_hidden] → [1, 256, text_hidden]
    // Squeeze batch dim for 2D matmul then unsqueeze back.
    let (b, seq, _) = normed.dims3()?;
    let flat = normed.reshape((b * seq, ()))?;
    let projected = flat.matmul(&vision.projection_weight)?.reshape((b, seq, ()))?;

    // L2 normalize along the last dimension
    let norm = projected.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let normalized = projected.broadcast_div(&(norm + 1e-6)?)?;

    Ok(normalized)
}

/// Simple RMS norm matching Gemma's style: x * (1 + weight) / sqrt(mean(x²) + eps)
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let hidden = x.dim(D::Minus1)?;
    let variance = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let normed = x_f32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let scale = (weight + 1.0)?;
    Ok(normed.to_dtype(x_dtype)?.broadcast_mul(&scale)?)
}

// ── Image analysis ──────────────────────────────────────────────────────────

// Special token IDs for Gemma 3 vision.
const BOI_TOKEN: u32 = 255999; // <start_of_image>
const EOI_TOKEN: u32 = 256000; // <end_of_image>

/// Build token sequence with image tokens inserted:
/// <start_of_turn>user\n<start_of_image>[img×256]<end_of_image>\n{text}<end_of_turn>\n<start_of_turn>model\n
fn build_vision_tokens(
    tokenizer: &tokenizers::Tokenizer,
    user_text: &str,
    vision: &VisionState,
) -> Result<(Vec<u32>, usize, usize)> {
    let text_enc = tokenizer
        .encode(user_text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let text_ids = text_enc.get_ids();
    let st = &vision.structural_tokens;

    let mut tokens = Vec::new();

    // <start_of_turn>user\n
    tokens.extend_from_slice(&st.start_turn);
    tokens.extend_from_slice(&st.user);
    tokens.extend_from_slice(&st.newline);

    // <start_of_image>[img×256]<end_of_image>\n
    tokens.push(BOI_TOKEN);
    let img_start = tokens.len();
    tokens.extend(std::iter::repeat_n(vision.image_token_index, vision.mm_tokens_per_image));
    let img_end = tokens.len();
    tokens.push(EOI_TOKEN);
    tokens.extend_from_slice(&st.newline);

    // {text}<end_of_turn>\n
    tokens.extend_from_slice(text_ids);
    tokens.extend_from_slice(&st.end_turn);
    tokens.extend_from_slice(&st.newline);

    // <start_of_turn>model\n
    tokens.extend_from_slice(&st.start_turn);
    tokens.extend_from_slice(&st.model);
    tokens.extend_from_slice(&st.newline);

    Ok((tokens, img_start, img_end))
}

/// Full vision+text pipeline for image analysis.
pub fn analyze_image(
    state: &mut LoadedState,
    vision: &VisionState,
    request: &ImageAnalysisRequest,
) -> Result<ImageAnalysisResponse> {
    let start = std::time::Instant::now();
    state.model.clear_kv_cache();

    // 1. Preprocess image
    let pixel_values = preprocess_image(
        &request.image_data,
        vision.vision_config.image_size,
        &state.device,
    )?;

    // 2. Encode image → [1, 256, text_hidden]
    let image_embeds = encode_image(vision, &pixel_values)?;

    // 3. Build token sequence with image token placeholders
    let (token_ids, img_start, img_end) = build_vision_tokens(
        &state.tokenizer,
        &request.prompt,
        vision,
    )?;

    // 4. Get text embeddings for all tokens
    let input_tensor = Tensor::new(&token_ids[..], &state.device)?.unsqueeze(0)?;
    let text_embeds = state.model.embed_tokens(&input_tensor)?;

    // 5. Replace image token embeddings with vision features
    let positions: Vec<usize> = (img_start..img_end).collect();
    let merged_embeds = replace_image_tokens(&text_embeds, &image_embeds, &positions)?;

    // 6. Forward with embeddings (prefill)
    let seq_len = merged_embeds.dim(1)?;
    let logits = state.model.forward_embeds(&merged_embeds, 0)?;
    let logits = logits.i((0, 0, ..))?;
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

    // 7. Autoregressive decode
    for i in 0..request.max_tokens.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &state.device)?.unsqueeze(0)?;
        let offset = seq_len + i + 1;
        let logits = state.model.forward(&input, offset)?;
        let logits = logits.i((0, 0, ..))?;
        next_token = sampler.sample(&logits)?;

        if state.eos_token_ids.contains(&next_token) {
            break;
        }
        generated.push(next_token);
    }

    let text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    let elapsed = start.elapsed().as_millis() as u64;
    Ok(ImageAnalysisResponse {
        text,
        tokens_generated: generated.len(),
        inference_time_ms: elapsed,
    })
}

/// Replace embeddings at image token positions with vision features.
fn replace_image_tokens(
    text_embeds: &Tensor,
    image_embeds: &Tensor,
    positions: &[usize],
) -> Result<Tensor> {
    let (_b, seq_len, _hidden) = text_embeds.dims3()?;
    let num_image_tokens = image_embeds.dim(1)?;

    if positions.len() != num_image_tokens {
        anyhow::bail!(
            "image token count mismatch: {} positions vs {} image embeddings",
            positions.len(),
            num_image_tokens,
        );
    }

    let mut parts: Vec<Tensor> = Vec::new();
    let block_start = positions[0];
    let block_end = positions[positions.len() - 1] + 1;

    if block_start > 0 {
        parts.push(text_embeds.i((0..1, ..block_start, ..))?);
    }
    parts.push(image_embeds.clone());
    if block_end < seq_len {
        parts.push(text_embeds.i((0..1, block_end..seq_len, ..))?);
    }

    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}
