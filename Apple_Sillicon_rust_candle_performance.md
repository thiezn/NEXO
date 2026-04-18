# BF16 + Candle + Apple Silicon

*   **The problem:** many open-weight LLMs ship as **BF16 safetensors**, but **Rust Candle + Metal** on Apple Silicon can still be awkward with BF16 depending on kernel/backend support. In practice, that can lead to casts/promotions that hurt speed or expand weights to **F32**, which increases memory use a lot. [\[linkedin.com\]](https://www.linkedin.com/pulse/demystifying-llm-quantization-gptq-awq-gguf-explained-xiao-fei-zhang-1lmbe/), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/README.md), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

*   **The tradeoff in Candle:**
    *   **BF16 safetensors** preserve the original checkpoint format, but may run into Metal/backend limitations. [\[linkedin.com\]](https://www.linkedin.com/pulse/demystifying-llm-quantization-gptq-awq-gguf-explained-xiao-fei-zhang-1lmbe/), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/README.md)
    *   **F32 fallback** is safer numerically, but roughly doubles weight memory vs BF16/F16. [\[github.com\]](https://github.com/huggingface/candle/issues/2891), [\[huggingface.co\]](https://huggingface.co/lmz/candle-quantized-phi/discussions/3)
    *   **F16** is usually better supported on Apple GPUs, but converting BF16 weights to F16 changes the numeric format and is not identical to the original checkpoint. [\[huggingface.co\]](https://huggingface.co/docs/diffusers/main/en/quantization/gguf), [\[huggingfac....github.io\]](https://huggingface.github.io/candle/)

*   **Pragmatic solution:** use a **quantized GGUF** model instead of raw BF16 safetensors. Quantization stores weights in compressed formats like **Q4 / Q5 / Q6 / Q8**, which reduces memory a lot and avoids relying on Candle’s plain BF16 tensor path. [\[stackoverflow.com\]](https://stackoverflow.com/questions/77359161/bfloat16-is-not-supported-on-mps-macos), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.1-quantization-formats-and-types)

*   **Why this helps on Apple Silicon:** Candle has **optimized quantized Metal kernels**, so GGUF weights stay in quantized form and are handled with **on-the-fly dequantization** during matmul, instead of loading the whole model as BF16 or F32 tensors. That means **less memory pressure**, no BF16 parsing headaches, and usually a much more practical runtime on Mac. [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/index.html), [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst)

*   **Mental model:**  
    **Raw BF16 safetensors = cleaner original weights, but awkward on Candle/Metal.**   
    **Quantized GGUF = slightly less precision, but much smaller and a better fit for Candle’s Metal backend.** [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/README.md), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs) [\[docs.rs\]](https://docs.rs/mlx-rust/latest/mlx_rust/), [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/index.html)

***

# Implementation plan: build safetensors first, then add GGUF

**first get normal weights working cleanly on Metal, then add GGUF, and end up with one codebase that supports both**. Candle already supports both **safetensors** and **GGUF/GGML** formats, but they use **different loading paths** internally, so the trick is to design your code so that only the **model-loading layer** changes while most of the rest stays shared. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models)

## Phase 1 — Build a clean safetensors path first

### Goal

Get a **baseline text-generation pipeline** working with normal weights on **Metal**, even if BF16 is not ideal yet. This lets you validate:

*   tokenizer behavior,
*   prompt formatting,
*   model config parsing,
*   generation loop,
*   KV-cache logic,
*   and output correctness,  
    before you add quantization complexity. [\[huggingface.co\]](https://huggingface.co/qwedsacf/gemma-4), [\[github.com\]](https://github.com/pytorch/pytorch/issues/141864)

### What code to implement now

Split your code into four layers:

1.  **Tokenizer layer**  
    Load `tokenizer.json` and handle encode/decode. Candle examples commonly use the `tokenizers` crate for this. Keeping tokenizer logic independent will make the GGUF transition easier later. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html)

2.  **Model config + loader layer**  
    This is where safetensors-specific loading lives. Candle’s normal model-loading path goes through **VarBuilder / safetensors backends**, which is separate from GGUF quantized loading. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[github.com\]](https://github.com/huggingface/candle/issues/2805)

3.  **Generation loop**  
    Keep prompt prefill, decode step, sampling, stop tokens, repetition penalty, and streaming output in a shared module. Candle’s examples reuse this style across many models. [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html), [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/)

4.  **Device/runtime setup**  
    Keep Metal device selection and runtime flags isolated. You want to test everything on **Metal from day one**, because CPU is too slow for iteration on Apple Silicon. Candle’s performance notes explicitly position Metal as the Apple GPU backend and recommend Metal + quantization for larger models on Apple hardware. [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/gguf-tokenizer.rs)

### End state of Phase 1

At the end of this phase, you should have one clean path:

```text
tokenizer.json + model config + safetensors -> Candle model -> shared generation loop
```

The important thing is: **the generation loop and tokenizer should already feel “finished.”** Only the model-loading/model-type layer should still be format-specific. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/)

***

## Phase 2 — Add GGUF as a second model backend

### What usually changes when switching from safetensors to GGUF?

This is the key question.

### Things that can stay the same

If you structure the code well, these parts should be mostly reusable:

*   **prompt formatting**,
*   **tokenization**,
*   **decode / sampling loop**,
*   **CLI / API surface**,
*   **streaming output**,
*   **chat history handling**,
*   **stop conditions**. [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html), [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/)

### Things that usually change

These are the parts that will typically differ:

1.  **Model loading**  
    Safetensors uses Candle’s standard tensor/VarBuilder loading path, while GGUF uses Candle’s **quantized GGUF loader** and quantized tensor types. Candle documents GGUF loading as a different path built around quantized tensors and GGUF metadata. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support)

2.  **Model type / weight structs**  
    For quantized models, Candle typically uses a **separate quantized model implementation**, not just “the same model with different weights.” For example, Candle has dedicated quantized model modules and examples such as **quantized-gemma** and other quantized LLMs. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[docs.rs\]](https://docs.rs/mlx-rust/latest/mlx_rust/)

3.  **Tokenizer source (optional)**  
    You can keep using `tokenizer.json` for both paths, which is the simplest option. But Candle now also has an example showing that a tokenizer can be built **directly from GGUF metadata**. That is optional, not required. [\[github.com\]](https://github.com/huggingface/candle/discussions/2941), [\[deepwiki.com\]](https://deepwiki.com/opendatalab/MinerU/8.5-apple-silicon-%28mpsmlx%29)

### Practical conclusion

The move from safetensors to GGUF is **not a full rewrite**, but it is also **not just “swap the file loader.”** The biggest change is usually the **model-loading/model-type layer**. The rest of the inference pipeline can stay mostly the same if you abstract it properly. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models), [\[docs.rs\]](https://docs.rs/mlx-rust/latest/mlx_rust/)

***

## Phase 3 — Design one interface that supports both

The cleanest structure is:

```rust
trait TextModel {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor>;
}
```

Then have two concrete implementations:

*   `SafetensorsGemmaModel`
*   `GgufQuantizedGemmaModel`

Both feed into the **same generation engine**. This mirrors the way Candle examples separate model construction from token generation logic. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html)

### Suggested project layout

```text
src/
  tokenizer.rs
  generation.rs
  model/
    mod.rs
    safetensors.rs
    gguf.rs
  main.rs
```

### Why this helps

It lets you do this:

*   start with `--format safetensors`,
*   test everything,
*   then add `--format gguf`,
*   while keeping the rest of the pipeline unchanged.

That is the best way to avoid duplicating generation code. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models)

***

## Phase 4 — Add GGUF carefully, with a minimal first target

### Recommended order

1.  **Keep the same tokenizer first** (`tokenizer.json`), even for GGUF. That avoids introducing two moving parts at once. GGUF tokenizer metadata support is nice, but not necessary for your first working version. [\[github.com\]](https://github.com/huggingface/candle/discussions/2941), [\[deepwiki.com\]](https://deepwiki.com/opendatalab/MinerU/8.5-apple-silicon-%28mpsmlx%29)
2.  **Load one known-good GGUF model** and verify prompt → output correctness. Candle’s quantized examples show this pattern for multiple models, including quantized Gemma examples. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[github.com\]](https://github.com/huggingface/candle/issues/1382)
3.  **Only after that**, consider optional niceties like tokenizer-from-GGUF metadata. [\[github.com\]](https://github.com/huggingface/candle/discussions/2941)

### Important warning

For GGUF, **architecture compatibility matters**. Candle examples and issues show that some GGUF files only work with the correct quantized model implementation, because metadata / architecture naming can differ. So don’t assume “any GGUF loads everywhere.” Pick a GGUF file that matches the Candle quantized model path you are targeting. [\[github.com\]](https://github.com/huggingface/candle/issues/2450), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.2-using-quantized-models)

***

## Recommended milestone checklist

### Milestone A — safetensors baseline

*   [ ] Metal device setup works. [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/gguf-tokenizer.rs)
*   [ ] Tokenizer encode/decode is correct. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html)
*   [ ] Generation loop produces sane output. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[doc.rust-lang.org\]](https://doc.rust-lang.org/stable/core/arch/x86/struct.bf16.html)
*   [ ] KV-cache and prompt prefill are stable. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[github.com\]](https://github.com/pytorch/pytorch/issues/141864)

### Milestone B — shared abstraction

*   [ ] `TextModel` trait (or equivalent enum wrapper) exists.
*   [ ] Generation loop no longer cares where weights came from.
*   [ ] CLI can switch model backend independently of tokenizer/generation settings.

### Milestone C — GGUF support

*   [ ] GGUF file loads through Candle quantized loader. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/5.2-using-quantized-models)
*   [ ] Quantized model produces correct text. [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[github.com\]](https://github.com/huggingface/candle/issues/1382)
*   [ ] Throughput/memory are measured against safetensors baseline. [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst)

***

# Bottom line

Your plan should be:

> **1. Implement safetensors first on Metal.** Confirm tokenizer, config, generation loop, and KV-cache all work.   
> **2. Keep tokenizer + generation shared.** Only isolate model-loading/model-type logic.   
> **3. Add GGUF as a second backend.** Expect moderate work in the loader/model layer, but not a full rewrite.   
> **4. End with one codebase that supports both**: normal safetensors for correctness/debugging, GGUF for practical Apple Silicon deployment. [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst) [\[openillumi.com\]](https://openillumi.com/en/en-pytorch-mps-bfloat16-error-fix-mac/), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models) [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4.3-using-quantized-models), [\[docs.rs\]](https://docs.rs/mlx-rust/latest/mlx_rust/) [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/7.2-model-format-support), [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/index.html), [\[ml-explore.github.io\]](https://ml-explore.github.io/mlx/build/html/_sources/install.rst)

If you want, I can turn this into a **very concrete Rust project skeleton** next — e.g. a `ModelBackend` enum / trait design and exactly which files/functions to create first.
