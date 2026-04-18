//! GGUF pipeline: load quantized Gemma 4 models and run chat/tool/image/audio inference.
//!
//! Text generation uses the quantized GGUF model. For image and audio analysis,
//! the vision/audio towers are lazily loaded from the mmproj GGUF on first use.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use super::text::QuantizedTextModel;
use crate::models::gemma4::config::{Gemma4Config, Gemma4VisionConfig};
use crate::models::gemma4::generation;
use crate::models::gemma4::safetensors::audio::AudioModel;
use crate::models::gemma4::safetensors::multimodal::broadcast_embed_to_mask;
use crate::models::gemma4::safetensors::multimodal_embedding::MultimodalEmbedder;
use crate::models::gemma4::safetensors::vision::VisionTower;
use crate::models::gemma4::template::Gemma4Template;
use crate::models::shared::weights::{find_gguf_file, load_gguf};
use crate::shared::templates::{ChatTemplate, ReasoningMode};
use crate::shared::types::{
    AudioAnalysisRequest, AudioAnalysisResponse, ChatRequest, ChatResponse, ImageAnalysisRequest,
    ImageAnalysisResponse, LayerKvSnapshot, ToolCallRequest, ToolCallResponse,
};

// ── Multimodal towers (lazily loaded from safetensors) ───────────────────

struct MultimodalTowers {
    vision_tower: VisionTower,
    embed_vision: MultimodalEmbedder,
    audio_tower: Option<AudioModel>,
    embed_audio: Option<MultimodalEmbedder>,
    vision_config: Gemma4VisionConfig,
    image_token_id: usize,
    audio_token_id: usize,
}

// ── Loaded state ─────────────────────────────────────────────────────────

pub struct LoadedState {
    model: QuantizedTextModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    stop_token_ids: Vec<u32>,
    max_context_tokens: usize,
    current_session_id: Option<String>,
    processed_tokens: Vec<u32>,
    /// Lazily loaded vision/audio towers from safetensors.
    multimodal: Option<MultimodalTowers>,
}

impl LoadedState {
    pub fn current_session_id(&self) -> Option<&str> {
        self.current_session_id.as_deref()
    }

    pub fn processed_tokens(&self) -> &[u32] {
        &self.processed_tokens
    }

    pub fn set_session_state(&mut self, session_id: Option<String>, tokens: Vec<u32>) {
        self.current_session_id = session_id;
        self.processed_tokens = tokens;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        DType::F32
    }

    pub fn kv_cache_seq_len(&self) -> usize {
        self.model.kv_cache_seq_len()
    }

    pub fn save_kv_cache(&self) -> candle_core::Result<Vec<LayerKvSnapshot>> {
        self.model.save_kv_cache()
    }

    pub fn restore_kv_cache(&mut self, snapshots: &[LayerKvSnapshot]) -> candle_core::Result<()> {
        self.model.restore_kv_cache(snapshots)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

// ── Load ─────────────────────────────────────────────────────────────────

/// Load a quantized Gemma 4 model from a GGUF file.
pub fn load(model_dir: &Path, max_context_tokens: Option<usize>) -> Result<LoadedState> {
    let start = Instant::now();

    let device = crate::device::create_device()?;
    tracing::info!("device ready in {:.1}s", start.elapsed().as_secs_f64());

    let gguf_path = find_gguf_file(model_dir, "", &["mmproj"])?;
    tracing::info!("loading GGUF: {}", gguf_path.display());

    let (content, mut file) = load_gguf(&gguf_path)?;

    if tracing::enabled!(tracing::Level::DEBUG) {
        for (k, v) in &content.metadata {
            let k: &str = k.as_str();
            if k.contains("block_count")
                || k.contains("head_count")
                || k.contains("embedding")
                || k.contains("sliding")
                || k.contains("rope")
                || k.contains("softcap")
                || k.contains("shared")
            {
                tracing::debug!("GGUF metadata: {} = {:?}", k, v);
            }
        }
    }

    let config_eos: u32 = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(1);

    let model = QuantizedTextModel::from_gguf(content, &mut file, &device)
        .map_err(|e| anyhow::anyhow!("failed to build quantized model: {e}"))?;

    tracing::info!("GGUF model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let stop_token_ids = generation::build_stop_token_ids(config_eos, &tokenizer);

    let max_ctx = max_context_tokens.unwrap_or(131072);

    tracing::info!(
        "GGUF Gemma 4 fully loaded in {:.1}s (stop_tokens={:?}, max_ctx={})",
        start.elapsed().as_secs_f64(),
        stop_token_ids,
        max_ctx,
    );

    Ok(LoadedState {
        model,
        tokenizer,
        device,
        stop_token_ids,
        max_context_tokens: max_ctx,
        current_session_id: None,
        processed_tokens: Vec::new(),
        multimodal: None,
    })
}

// ── Chat ─────────────────────────────────────────────────────────────────

pub fn chat(state: &mut LoadedState, request: &ChatRequest) -> Result<ChatResponse> {
    let template = Gemma4Template;
    let prompt = template.format_prompt(&request.messages, &ReasoningMode::Disabled);
    tracing::trace!("chat prompt ({} chars): '{:.200}'", prompt.len(), prompt);

    let (new_tokens, prefix_len) = generation::tokenize_with_prefix(
        &state.tokenizer,
        &prompt,
        &state.processed_tokens,
        state.current_session_id.as_deref(),
        &request.session_id,
    )?;

    let (raw, tokens_generated, inference_time_ms) = generation::generate(
        &mut state.model,
        &state.tokenizer,
        &state.device,
        &state.stop_token_ids,
        state.max_context_tokens,
        new_tokens.clone(),
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        prefix_len,
    )?;

    state.current_session_id = request.session_id.clone();
    state.processed_tokens = new_tokens;

    let text = template.parse_response(&raw);

    Ok(ChatResponse {
        text,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Tool calling ─────────────────────────────────────────────────────────

pub fn call_tools(state: &mut LoadedState, request: &ToolCallRequest) -> Result<ToolCallResponse> {
    let template = Gemma4Template;
    let prompt =
        template.format_with_tools(&request.messages, &request.tools, &ReasoningMode::Disabled);
    tracing::trace!(
        "tool_call prompt ({} chars, {} tools): '{:.200}'",
        prompt.len(),
        request.tools.len(),
        prompt,
    );

    let (new_tokens, prefix_len) = generation::tokenize_with_prefix(
        &state.tokenizer,
        &prompt,
        &state.processed_tokens,
        state.current_session_id.as_deref(),
        &request.session_id,
    )?;

    let (raw, tokens_generated, inference_time_ms) = generation::generate(
        &mut state.model,
        &state.tokenizer,
        &state.device,
        &state.stop_token_ids,
        state.max_context_tokens,
        new_tokens.clone(),
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        prefix_len,
    )?;

    state.current_session_id = request.session_id.clone();
    state.processed_tokens = new_tokens;

    let (tool_calls, reasoning) = template.parse_tool_calls(&raw);

    Ok(ToolCallResponse {
        tool_calls,
        reasoning,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Multimodal upgrade ───────────────────────────────────────────────────

/// Lazily load vision/audio towers from the mmproj GGUF on first multimodal request.
fn ensure_multimodal(state: &mut LoadedState, model_dir: &Path) -> Result<()> {
    if state.multimodal.is_some() {
        return Ok(());
    }

    tracing::info!("loading vision/audio towers from mmproj GGUF...");
    let start = Instant::now();

    // Load config.json for architecture parameters
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("config.json needed for multimodal — {e}"))?;
    let config: Gemma4Config = serde_json::from_str(&config_str)?;

    // Find and load the mmproj GGUF file
    let mmproj_path = find_gguf_file(model_dir, "mmproj", &[])?;
    tracing::info!("loading mmproj: {}", mmproj_path.display());

    let vis_dtype = crate::device::gpu_compute_dtype(&state.device);
    let tensors = super::mmproj::load_mmproj_tensors(&mmproj_path, &state.device)?;

    // Vision tower uses the GPU compute dtype (BF16 on Metal).
    let vb_vis = VarBuilder::from_tensors(tensors.clone(), vis_dtype, &state.device);
    let vision_tower = VisionTower::new(&config.vision_config, vb_vis.pp("vision_tower"))
        .map_err(|e| anyhow::anyhow!("failed to load vision tower: {e}"))?;

    let vis_hidden = config.vision_config.hidden_size;
    let text_hidden = config.text_config.hidden_size;
    let embed_vision = MultimodalEmbedder::new(
        vis_hidden,
        text_hidden,
        config.vision_config.rms_norm_eps,
        vb_vis.pp("embed_vision"),
    )
    .map_err(|e| anyhow::anyhow!("failed to load vision embedder: {e}"))?;

    // Audio tower performs attention in F32 internally, so weights must be F32
    // to avoid dtype mismatches in matmul.
    let (audio_tower, embed_audio) = if let Some(ref audio_cfg) = config.audio_config {
        let vb_audio = VarBuilder::from_tensors(tensors, DType::F32, &state.device);
        let tower = AudioModel::new(audio_cfg, vb_audio.pp("audio_tower"))
            .map_err(|e| anyhow::anyhow!("failed to load audio tower: {e}"))?;
        let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
        let embed = MultimodalEmbedder::new(
            audio_hidden,
            text_hidden,
            audio_cfg.rms_norm_eps,
            vb_audio.pp("embed_audio"),
        )
        .map_err(|e| anyhow::anyhow!("failed to load audio embedder: {e}"))?;
        (Some(tower), Some(embed))
    } else {
        (None, None)
    };

    tracing::info!(
        "multimodal towers loaded from mmproj in {:.1}s (audio={})",
        start.elapsed().as_secs_f64(),
        audio_tower.is_some(),
    );

    state.multimodal = Some(MultimodalTowers {
        vision_tower,
        embed_vision,
        audio_tower,
        embed_audio,
        vision_config: config.vision_config,
        image_token_id: config.image_token_id,
        audio_token_id: config.audio_token_id,
    });

    Ok(())
}

// ── Image analysis ───────────────────────────────────────────────────────

/// Analyze an image using the GGUF text model + safetensors vision tower.
pub fn analyze_image(
    state: &mut LoadedState,
    request: &ImageAnalysisRequest,
    model_dir: &Path,
) -> Result<ImageAnalysisResponse> {
    ensure_multimodal(state, model_dir)?;

    let mm = state
        .multimodal
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("multimodal towers not loaded"))?;
    let vis_cfg = mm.vision_config.clone();

    // Decode and preprocess image
    let image = image::load_from_memory(&request.image_data)
        .map_err(|e| anyhow::anyhow!("failed to decode image: {e}"))?;
    let dtype = crate::device::gpu_compute_dtype(&state.device);
    let (pixel_values, img_h, img_w) =
        crate::models::gemma4::safetensors::pipeline::preprocess_image(
            &image,
            &vis_cfg,
            &state.device,
            dtype,
        )?;

    let num_image_tokens = crate::models::gemma4::safetensors::pipeline::compute_num_image_tokens(
        img_h,
        img_w,
        vis_cfg.patch_size as u32,
        vis_cfg.pooling_kernel_size as u32,
    );
    tracing::debug!(
        "image {}x{} -> {} image tokens",
        img_w,
        img_h,
        num_image_tokens,
    );

    // Build prompt with image placeholders
    let image_placeholders = "<|image|>".repeat(num_image_tokens);
    let prompt = format!(
        "<|turn>user\n<|image>{image_placeholders}<image|>\n{}<turn|>\n<|turn>model\n",
        request.prompt
    );

    let (text, tokens_generated, inference_time_ms) = multimodal_generate(
        state,
        &prompt,
        request.temperature,
        request.max_tokens,
        |s, input| {
            multimodal_forward(
                s,
                input,
                Some(std::slice::from_ref(&pixel_values)),
                None,
                None,
            )
        },
    )?;

    tracing::info!("GGUF image analysis: {tokens_generated} tokens in {inference_time_ms}ms");

    Ok(ImageAnalysisResponse {
        text,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Audio analysis ───────────────────────────────────────────────────────

/// Analyze audio using the GGUF text model + safetensors audio tower.
pub fn analyze_audio(
    state: &mut LoadedState,
    request: &AudioAnalysisRequest,
    model_dir: &Path,
) -> Result<AudioAnalysisResponse> {
    ensure_multimodal(state, model_dir)?;

    let mm = state
        .multimodal
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("multimodal towers not loaded"))?;
    if mm.audio_tower.is_none() {
        anyhow::bail!("this model does not have an audio tower");
    }

    // Audio tower runs in F32 (its attention and convolutions require it).
    let (mel_tensor, mask_tensor, num_mel_frames) =
        crate::models::gemma4::safetensors::pipeline::prepare_audio(
            &request.pcm_samples,
            request.sample_rate,
            &state.device,
            DType::F32,
        )?;

    let num_audio_tokens =
        crate::models::gemma4::safetensors::pipeline::compute_num_audio_tokens(num_mel_frames, 1);
    tracing::debug!(
        "audio: {} samples @ {}Hz -> {} mel frames -> {} audio tokens",
        request.pcm_samples.len(),
        request.sample_rate,
        num_mel_frames,
        num_audio_tokens,
    );

    let audio_placeholders = "<|audio|>".repeat(num_audio_tokens);
    let prompt = format!(
        "<|turn>user\n<|audio>{audio_placeholders}<audio|>\n{}<turn|>\n<|turn>model\n",
        request.prompt
    );

    let (text, tokens_generated, inference_time_ms) = multimodal_generate(
        state,
        &prompt,
        request.temperature,
        request.max_tokens,
        |s, input| multimodal_forward(s, input, None, Some(&mel_tensor), Some(&mask_tensor)),
    )?;

    tracing::info!("GGUF audio analysis: {tokens_generated} tokens in {inference_time_ms}ms");

    Ok(AudioAnalysisResponse {
        text,
        tokens_generated,
        inference_time_ms,
    })
}

// ── Multimodal generation loop ──────────────────────────────────────────

/// Shared token generation loop for multimodal analysis (image/audio).
///
/// `first_token_forward` is called on the first token with multimodal embeddings;
/// subsequent tokens use text-only forward through the GGUF model.
fn multimodal_generate(
    state: &mut LoadedState,
    prompt: &str,
    temperature: f64,
    max_tokens: usize,
    first_token_forward: impl FnOnce(&mut LoadedState, &Tensor) -> Result<Tensor>,
) -> Result<(String, usize, u64)> {
    let start = Instant::now();

    let encoding = state
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
    let mut tokens = encoding.get_ids().to_vec();

    let sampling = if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: 0.95,
            temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

    state.model.clear_kv_cache();

    let mut generated_tokens = 0usize;
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut first_forward = Some(first_token_forward);

    for _index in 0..max_tokens {
        let context_size = if first_forward.is_some() {
            tokens.len()
        } else {
            1
        };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &state.device)?.unsqueeze(0)?;

        let logits = if let Some(fwd) = first_forward.take() {
            fwd(state, &input)?
        } else {
            let embeds = state.model.embed_tokens(&input)?;
            let pli = state.model.compute_per_layer_inputs(&input, &embeds)?;
            state
                .model
                .forward_embeds(&embeds, start_pos, pli.as_ref())?
        };

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let logits = if !tokens.is_empty() {
            let penalty_start = tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                1.1,
                &tokens[penalty_start..],
            )?
        } else {
            logits
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if state.stop_token_ids.contains(&next_token) {
            break;
        }

        output_tokens.push(next_token);
    }

    let raw = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))?;

    let template = Gemma4Template;
    let text = template.parse_response(&raw);
    let inference_time_ms = start.elapsed().as_millis() as u64;

    Ok((text, generated_tokens, inference_time_ms))
}

// ── Multimodal forward (shared between image and audio) ──────────────────

/// Run a multimodal forward pass: embed tokens, inject vision/audio embeddings,
/// then forward through the quantized text model.
fn multimodal_forward(
    state: &mut LoadedState,
    input_ids: &Tensor,
    pixel_values: Option<&[Tensor]>,
    audio_mel: Option<&Tensor>,
    audio_mel_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let mut input_embeds = state.model.embed_tokens(input_ids)?;
    let per_layer_inputs = state
        .model
        .compute_per_layer_inputs(input_ids, &input_embeds)
        .map_err(|e| anyhow::anyhow!("PLE computation failed: {e}"))?;

    let mm = state
        .multimodal
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("multimodal towers not loaded"))?;

    if let Some(pixel_values) = pixel_values {
        let image_mask = input_ids
            .to_dtype(DType::F32)?
            .eq(mm.image_token_id as f64)?;

        let vision_features = mm
            .vision_tower
            .forward(pixel_values)
            .map_err(|e| anyhow::anyhow!("vision tower forward failed: {e}"))?;
        let image_embeds = mm
            .embed_vision
            .forward(&vision_features)?
            .to_dtype(input_embeds.dtype())?;

        let image_embeds_flat = image_embeds.squeeze(0)?;
        let mask_expanded = image_mask
            .unsqueeze(candle_core::D::Minus1)?
            .broadcast_as(input_embeds.shape())?
            .to_dtype(input_embeds.dtype())?;
        let image_embeds_broadcast = broadcast_embed_to_mask(&image_embeds_flat, &image_mask)
            .map_err(|e| anyhow::anyhow!("broadcast embed to mask failed: {e}"))?;
        input_embeds = ((mask_expanded.clone() * image_embeds_broadcast)?
            + ((1.0 - mask_expanded)? * input_embeds)?)?;
    }

    if let (Some(audio_mel), Some(audio_mel_mask), Some(audio_tower), Some(embed_audio)) =
        (audio_mel, audio_mel_mask, &mm.audio_tower, &mm.embed_audio)
    {
        let audio_mask = input_ids
            .to_dtype(DType::F32)?
            .eq(mm.audio_token_id as f64)?;

        let (audio_features, enc_mask) = audio_tower
            .forward(audio_mel, audio_mel_mask)
            .map_err(|e| anyhow::anyhow!("audio tower forward failed: {e}"))?;
        let valid = enc_mask.eq(0.0)?;
        let batch = audio_features.dim(0)?;
        let mut all_feats = Vec::new();
        for b in 0..batch {
            let valid_b = valid.get(b)?;
            let valid_sum = valid_b
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()? as usize;
            if valid_sum > 0 {
                all_feats.push(audio_features.get(b)?.narrow(0, 0, valid_sum)?);
            }
        }
        if !all_feats.is_empty() {
            let audio_feats = Tensor::cat(&all_feats, 0)?.unsqueeze(0)?;
            let audio_embeds = embed_audio
                .forward(&audio_feats)?
                .to_dtype(input_embeds.dtype())?;

            let audio_embeds_flat = audio_embeds.squeeze(0)?;
            let mask_expanded = audio_mask
                .unsqueeze(candle_core::D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(input_embeds.dtype())?;
            let audio_embeds_broadcast =
                broadcast_embed_to_mask(&audio_embeds_flat, &audio_mask)
                    .map_err(|e| anyhow::anyhow!("broadcast embed to mask failed: {e}"))?;
            input_embeds = ((mask_expanded.clone() * audio_embeds_broadcast)?
                + ((1.0 - mask_expanded)? * input_embeds)?)?;
        }
    }

    state
        .model
        .forward_embeds(&input_embeds, 0, per_layer_inputs.as_ref())
        .map_err(|e| anyhow::anyhow!("GGUF forward_embeds failed: {e}"))
}
