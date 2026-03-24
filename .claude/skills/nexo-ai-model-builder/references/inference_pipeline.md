# Inference Pipeline Guide

## Check candle-transformers first

Before implementing model architectures from scratch, check if `candle-transformers` already includes them. As of v0.9.2, it includes: `parler_tts`, `dac`, `snac`, `whisper`, `t5`, `clip`, `flux`, `llama`, `qwen3`, `mimi`, `encodec`, `metavoice`, `csm`, and many more. Use `candle_transformers::models::<model_name>` directly.

## Device & Dtype

```rust
use crate::device;

// Auto-select Metal on macOS, CPU as fallback
let device = device::create_device()?;

// F32 on Metal/CPU (BF16 has precision issues on Metal)
let dtype = device::gpu_dtype(&device);

// Check memory before loading
device::preflight_memory_check(model_name, estimated_bytes)?;
```

Always create the device once and pass it to both preprocessing and model loading. Creating multiple devices wastes memory and can cause Metal context issues.

## Model Loading

### Single safetensors file

```rust
use candle_core::{Device, DType};
use candle_nn::VarBuilder;

let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)?
};
let model = ModelType::new(&config, vb)?;
```

### Sharded safetensors

```rust
let paths: Vec<PathBuf> = vec![shard_0, shard_1, shard_2];
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)?
};
```

### LoadedState pattern

```rust
struct LoadedState {
    model: ModelType,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

// In load():
let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
let model = ModelType::new(&config, vb)?;
self.state = Some(LoadedState { model, tokenizer, config, device });
```

## Tokenizer & Chat Templates

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!(e))?;

// Encode prompt
let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
```

### Chat template assembly (for chat/tool models)

Build input_ids as `Vec<i64>` directly to avoid type conversion overhead:

```rust
fn build_input_ids(tokenizer: &Tokenizer, prompt: &str, system_prompt: &str) -> Vec<i64> {
    let mut ids = Vec::new();
    // <|im_start|>system\n{system_prompt}<|im_end|>\n
    ids.extend(encode_special(tokenizer, "<|im_start|>"));
    ids.extend(encode_text(tokenizer, &format!("system\n{system_prompt}")));
    ids.extend(encode_special(tokenizer, "<|im_end|>"));
    ids.push(newline_id);
    // <|im_start|>user\n{prompt}<|im_end|>\n
    // ...
    // <|im_start|>assistant\n
    ids
}
```

## Forward Pass

### Autoregressive text generation (Chat, Tool, Image analysis)

```rust
fn generate(state: &mut LoadedState, input_ids: &[i64], max_tokens: usize, temperature: f64, top_p: f64) -> Result<(String, usize)> {
    let device = &state.device;
    let mut tokens = input_ids.to_vec();
    let mut generated = Vec::with_capacity(max_tokens);

    // Prefill: process full input sequence
    let input_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = state.model.forward(&input_tensor, 0)?;
    let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
    let last_logits = logits.get(logits.dim(0)? - 1)?;
    let mut next_token = sample_token(&last_logits, temperature, top_p)?;

    // Decode: one token at a time
    for pos in input_ids.len()..input_ids.len() + max_tokens {
        if is_eos(next_token, &state.config) { break; }
        generated.push(next_token as u32);

        let token_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = state.model.forward(&token_tensor, pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        next_token = sample_token(&logits, temperature, top_p)?;
    }

    let text = state.tokenizer.decode(&generated, true).map_err(|e| anyhow::anyhow!(e))?;
    Ok((text, generated.len()))
}
```

### Non-autoregressive (Listen, Talk, Imagine)

These models typically have a single forward pass or a fixed number of steps:

```rust
// Whisper-style transcription
let mel = preprocess_audio(&samples, sample_rate)?;
let tokens = state.model.forward(&mel)?;
let text = state.tokenizer.decode(&tokens, true)?;

// Diffusion image generation (sequential load-use-drop for VRAM)
let embeddings = {
    let encoder = load_text_encoder(&paths, &device, dtype)?;
    encoder.encode(&prompt)?
};
// Encoder dropped here, freeing VRAM
let mut model = load_unet(&paths, &device, dtype)?;
for step in 0..num_steps {
    latents = model.step(&latents, &embeddings, step)?;
}
let image = vae_decode(&latents)?;
```

## Token Sampling

```rust
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<i64> {
    if temperature == 0.0 {
        // Greedy: argmax
        let token = logits.argmax(0)?.to_scalar::<u32>()? as i64;
        return Ok(token);
    }

    // Temperature scaling
    let logits = (logits / temperature)?;
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Top-p nucleus sampling
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0;
    let mut candidates = Vec::new();
    for (idx, p) in indexed {
        cumsum += p;
        candidates.push((idx, p));
        if cumsum >= top_p as f32 { break; }
    }

    // Sample from candidates
    use rand::distr::weighted::WeightedIndex;
    use rand::prelude::*;

    let weights: Vec<f32> = candidates.iter().map(|(_, p)| *p).collect();
    let dist = WeightedIndex::new(&weights)?;
    let mut rng = rand::rng();
    let sampled = candidates[dist.sample(&mut rng)].0;
    Ok(sampled as i64)
}
```

## Noise Generation (for diffusion models)

Always use CPU-seeded RNG, then move to device. Never use `device.set_seed()` + `Tensor::randn()` — it's non-deterministic on Metal:

```rust
fn seeded_randn(seed: u64, shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand_distr::StandardNormal;
    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|_| rng.sample(normal))
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu)?
        .to_device(device)?
        .to_dtype(dtype)
}
```

## Sequential load-use-drop (multi-component pipelines)

When VRAM is limited and the pipeline has separable stages (e.g. text encoder → diffusion model), load each component, use it, and drop it before loading the next:

```rust
let embeddings = {
    let encoder = load_text_encoder(&paths, &device, dtype)?;
    encoder.encode(&prompt)?
};
// encoder is dropped here, freeing VRAM for the main model
let model = load_unet(&paths, &device, dtype)?;
```

This isn't needed for autoregressive models where the whole model must stay loaded during generation.

## Vision-Language Models

### Image preprocessing

1. **Smart resize**: find (H, W) divisible by `patch_size * spatial_merge_size`, keeping total pixels within min/max bounds and preserving aspect ratio
2. **Normalize + patchify in a single fused pass**: iterate over patches directly, normalizing pixel values inline
3. **Temporal frame duplication**: for models with `temporal_patch_size > 1`, write the same normalized value into each temporal slot
4. Return `PreprocessedInput { pixel_values: Tensor, grid_thw: Tensor, num_image_tokens: usize }`

### Continuous spans for vision embedding

Vision models need to know where image tokens are in the input sequence:

```rust
fn find_continuous_spans(input_ids: &[i64], token_id: i64) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, &id) in input_ids.iter().enumerate() {
        if id == token_id {
            if start.is_none() { start = Some(i); }
        } else if let Some(s) = start.take() {
            spans.push((s, i));
        }
    }
    if let Some(s) = start { spans.push((s, input_ids.len())); }
    spans
}
```

## Metal Considerations

- Use F32 dtype on Metal (BF16 has precision issues)
- Create device once and pass it everywhere
- Use `preflight_memory_check()` before loading large models
- Unified memory means GPU and CPU share the same physical RAM
- Drop models explicitly to free memory before loading the next one
