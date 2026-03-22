use anyhow::Result;
use local_inference_helpers::candle_core::{DType, Device, Tensor};
use local_inference_helpers::candle_nn::VarBuilder;
use candle_transformers::models::t5;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::t5_gguf::GgufT5Encoder;

/// T5-XXL config (hardcoded for FLUX).
pub fn config() -> t5::Config {
    t5::Config {
        vocab_size: 32128,
        d_model: 4096,
        d_kv: 64,
        d_ff: 10240,
        num_heads: 64,
        num_layers: 24,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.1,
        layer_norm_epsilon: 1e-6,
        initializer_factor: 1.0,
        feed_forward_proj: t5::ActivationWithOptionalGating {
            gated: true,
            activation: local_inference_helpers::candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: false,
        use_cache: true,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: Some(0),
        is_decoder: false,
        is_encoder_decoder: true,
        num_decoder_layers: Some(24),
    }
}

pub(crate) enum T5Model {
    FP16(t5::T5EncoderModel),
    Quantized(GgufT5Encoder),
}

impl T5Model {
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::FP16(m) => Ok(m.forward(input_ids)?),
            Self::Quantized(m) => m.forward(input_ids),
        }
    }
}

pub(crate) struct T5Encoder {
    pub model: Option<T5Model>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub is_quantized: bool,
}

impl T5Encoder {
    pub fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let is_quantized = encoder_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        let model = if is_quantized {
            T5Model::Quantized(GgufT5Encoder::load(encoder_path, device)?)
        } else {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(encoder_path),
                    dtype,
                    device,
                )?
            };
            T5Model::FP16(t5::T5EncoderModel::load(vb, &config())?)
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load T5 tokenizer: {e}"))?;

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            is_quantized,
        })
    }

    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let t5 = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("T5 model unavailable"))?;

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        tokens.resize(256, 0);

        let input_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let emb = t5.forward(&input_ids)?;
        Ok(emb.to_device(target_device)?.to_dtype(target_dtype)?)
    }

    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    pub fn reload(&mut self, encoder_path: &PathBuf, dtype: DType) -> Result<()> {
        if self.is_quantized {
            self.model = Some(T5Model::Quantized(GgufT5Encoder::load(
                encoder_path,
                &self.device,
            )?));
        } else {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(encoder_path),
                    dtype,
                    &self.device,
                )?
            };
            self.model = Some(T5Model::FP16(t5::T5EncoderModel::load(vb, &config())?));
        }
        Ok(())
    }
}
