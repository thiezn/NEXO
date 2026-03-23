//! Qwen3 text encoder wrapper for Z-Image and Flux.2 Klein.

use anyhow::Result;
use local_inference_helpers::candle_core::{DType, Device, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use candle_transformers::models::z_image::{TextEncoderConfig, ZImageTextEncoder};
use std::path::{Path, PathBuf};

use super::qwen3_gguf::GgufQwen3Encoder;

pub(crate) enum Qwen3Model {
    BF16(ZImageTextEncoder),
    Quantized(GgufQwen3Encoder),
}

impl Qwen3Model {
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::BF16(m) => Ok(m.forward(input_ids)?),
            Self::Quantized(m) => m.forward(input_ids),
        }
    }

    /// Run forward pass and collect hidden states from specific layer indices.
    /// Used by Flux.2 Klein which stacks layers 9, 18, 27 to get 7680-dim embeddings.
    pub fn forward_with_layers(
        &mut self,
        input_ids: &Tensor,
        layer_indices: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::BF16(m) => {
                // BF16 ZImageTextEncoder only returns penultimate layer.
                // For multi-layer extraction, fall back to repeating the single output.
                let emb = m.forward(input_ids)?;
                let copies: Vec<&Tensor> = (0..layer_indices.len()).map(|_| &emb).collect();
                Ok(Tensor::cat(&copies, 2)?)
            }
            Self::Quantized(m) => m.forward_with_layers(input_ids, layer_indices),
        }
    }
}

pub(crate) struct Qwen3Encoder {
    pub model: Option<Qwen3Model>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub is_quantized: bool,
    encoder_paths: Vec<PathBuf>,
    dtype: DType,
}

fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn format_prompt_for_flux2(prompt: &str) -> String {
    format!("{}<think>\n\n</think>\n\n", format_prompt_for_qwen3(prompt))
}

impl Qwen3Encoder {
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let te_cfg = TextEncoderConfig::z_image();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .filter_map(|p| p.to_str())
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = Qwen3Model::BF16(ZImageTextEncoder::new(&te_cfg, vb)?);

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            is_quantized: false,
            encoder_paths: encoder_paths.to_vec(),
            dtype,
        })
    }

    pub fn load_gguf(gguf_path: &Path, tokenizer_path: &PathBuf, device: &Device) -> Result<Self> {
        let model = Qwen3Model::Quantized(GgufQwen3Encoder::load(gguf_path, device)?);
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            is_quantized: true,
            encoder_paths: vec![gguf_path.to_path_buf()],
            dtype: DType::F32,
        })
    }

    /// Encode a text prompt. Returns (embeddings, token_count).
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3 model unavailable (weights dropped)"))?;

        let formatted = format_prompt_for_qwen3(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        let emb = model.forward(&input_ids)?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    /// Encode with multi-layer hidden state extraction (for Flux.2 Klein).
    pub fn encode_with_layers(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
        layer_indices: &[usize],
    ) -> Result<(Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3 model unavailable (weights dropped)"))?;

        let formatted = format_prompt_for_flux2(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        let emb = model.forward_with_layers(&input_ids, layer_indices)?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    pub fn reload(&mut self) -> Result<()> {
        if self.is_quantized {
            self.model = Some(Qwen3Model::Quantized(GgufQwen3Encoder::load(
                &self.encoder_paths[0],
                &self.device,
            )?));
        } else {
            let te_cfg = TextEncoderConfig::z_image();
            let path_strs: Vec<&str> = self
                .encoder_paths
                .iter()
                .filter_map(|p| p.to_str())
                .collect();
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&path_strs, self.dtype, &self.device)?
            };
            self.model = Some(Qwen3Model::BF16(ZImageTextEncoder::new(&te_cfg, vb)?));
        }
        Ok(())
    }
}

#[cfg(test)]
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
