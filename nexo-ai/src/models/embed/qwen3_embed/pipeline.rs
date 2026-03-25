use std::path::Path;

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};

use crate::models::multipurpose::qwen3::qwen3_dense;
use crate::shared::types::*;

pub struct LoadedState {
    pub weights: qwen3_dense::ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: candle_core::Device,
}

pub fn load(model_dir: &Path) -> Result<LoadedState> {
    let device = crate::device::create_device(|msg| tracing::info!("{msg}"))?;

    let gguf_path = crate::models::shared::weights::find_gguf_file(model_dir, "", &[])?;
    tracing::info!("loading GGUF embedding model from {}", gguf_path.display());

    let (content, mut file) = crate::models::shared::weights::load_gguf(&gguf_path)?;
    let weights = qwen3_dense::ModelWeights::from_gguf(content, &mut file, &device)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    tracing::info!("qwen3 embedding model loaded");

    Ok(LoadedState {
        weights,
        tokenizer,
        device,
    })
}

/// Generate embeddings for a batch of texts.
pub fn embed(state: &mut LoadedState, request: &EmbedRequest) -> Result<EmbedResponse> {
    let start = std::time::Instant::now();

    let mut all_embeddings = Vec::with_capacity(request.texts.len());
    let mut dimensions = 0;

    for text in &request.texts {
        let encoding = state
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = token_ids.len();

        state.weights.clear_kv_cache();
        let input = Tensor::new(&token_ids[..], &state.device)?.unsqueeze(0)?;

        // Forward pass to get hidden states for all positions
        let hidden = state.weights.forward_hidden(&input, 0)?;

        // Last-token pooling: take the hidden state of the last token
        let last_hidden = hidden.i((.., seq_len - 1, ..))?.squeeze(0)?;

        // L2 normalize
        let embedding = l2_normalize(&last_hidden)?;
        let emb_vec: Vec<f32> = embedding.to_vec1()?;
        dimensions = emb_vec.len();
        all_embeddings.push(emb_vec);
    }

    let inference_time_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        "embedded {} text(s) in {:.1}s (dim={})",
        request.texts.len(),
        inference_time_ms as f64 / 1000.0,
        dimensions
    );

    Ok(EmbedResponse {
        embeddings: all_embeddings,
        dimensions,
        inference_time_ms,
    })
}

/// L2-normalize a 1D tensor.
fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norm = tensor
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    if norm < 1e-12 {
        return Ok(tensor.clone());
    }
    Ok((tensor / norm as f64)?)
}
