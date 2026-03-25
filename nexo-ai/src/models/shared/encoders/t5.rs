use anyhow::Result;
use candle_core::{Device, Tensor};

/// Tokenize text and return input_ids as a batched tensor `[1, seq_len]`.
///
/// This is a common operation for models that use a text encoder (e.g. Parler-TTS,
/// FLUX). The tokenizer should already be loaded from a `tokenizer.json` file.
pub fn encode_text(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    let ids = Tensor::from_vec(token_ids, (1, seq_len), device)?;
    Ok(ids)
}
