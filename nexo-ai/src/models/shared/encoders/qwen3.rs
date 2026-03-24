//! Qwen3 text encoder for Flux.2 image generation.
//!
//! Encodes text prompts into hidden-state embeddings that condition
//! the Flux.2 transformer. Uses `candle_transformers::models::z_image`
//! for the BF16 model. Multi-layer extraction (layers 9, 18, 27) is
//! approximated by repeating the penultimate-layer output.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{TextEncoderConfig, ZImageTextEncoder};
use std::path::Path;

pub struct Qwen3Encoder {
    model: ZImageTextEncoder,
    tokenizer: tokenizers::Tokenizer,
}

fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn format_prompt_for_flux2(prompt: &str) -> String {
    format!("{}<think>\n\n</think>\n\n", format_prompt_for_qwen3(prompt))
}

impl Qwen3Encoder {
    pub fn load(
        encoder_paths: &[impl AsRef<Path>],
        tokenizer_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let te_cfg = TextEncoderConfig::z_image();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .filter_map(|p| p.as_ref().to_str())
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = ZImageTextEncoder::new(&te_cfg, vb)?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        Ok(Self { model, tokenizer })
    }

    /// Encode with multi-layer hidden state extraction (for Flux.2).
    ///
    /// Stacks `layer_indices.len()` copies of the penultimate-layer output
    /// along dim 2 to produce the expected joint_attention_dim.
    pub fn encode_with_layers(
        &self,
        prompt: &str,
        device: &Device,
        dtype: DType,
        layer_indices: &[usize],
    ) -> Result<(Tensor, usize)> {
        let formatted = format_prompt_for_flux2(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), device)?;

        // BF16 ZImageTextEncoder returns penultimate layer only.
        // Stack n copies to approximate multi-layer extraction.
        let emb = self.model.forward(&input_ids)?;
        let copies: Vec<&Tensor> = (0..layer_indices.len()).map(|_| &emb).collect();
        let stacked = Tensor::cat(&copies, 2)?;

        let stacked = stacked.to_device(device)?.to_dtype(dtype)?;
        Ok((stacked, token_count))
    }

}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn z_image_chat_template() {
        let result = format_prompt_for_qwen3("a cat");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a cat"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
        assert!(!result.contains("<think>"));
    }

    #[test]
    fn flux2_chat_template_includes_thinking() {
        let result = format_prompt_for_flux2("a sunset");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a sunset"));
        assert!(result.contains("<think>\n\n</think>\n\n"));
        assert!(result.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn templates_differ_only_in_thinking_block() {
        let z = format_prompt_for_qwen3("test");
        let f = format_prompt_for_flux2("test");
        assert_eq!(f, format!("{z}<think>\n\n</think>\n\n"));
    }

    #[test]
    fn templates_exact_structure() {
        assert_eq!(
            format_prompt_for_qwen3("hello"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            format_prompt_for_flux2("hello"),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
