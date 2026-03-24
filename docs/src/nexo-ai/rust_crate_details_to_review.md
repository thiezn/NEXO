## My Rust Crate approach

Below is the **best Rust-first workflow I would recommend for your constraints**: **Rust/Candle at inference time**, your own data/labeling pipeline, and **LoRA + QLoRA training in Rust where possible**, with a clean path to **Candle-consumable artifacts**. I’m optimizing for **(1) artifact compatibility with Candle, (2) lowest integration risk, and (3) enough flexibility that you can extend the stack yourself**. [\[github.com\]](https://github.com/huggingface/candle), [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs)

***

# Executive recommendation

## My short list

### 1) **Tool-calling LoRA (LLMs): use `peft-rs` as your primary training crate**

`peft-rs` is the most promising Rust-native **general PEFT** library I found for LLM-style adaptation: it is **pure Rust**, built **on Candle**, supports **LoRA/DoRA/AdaLoRA/IA³/LoHa/LoKr/OFT/BOFT/VeRA/prefix/prompt tuning**, exposes **training utilities**, and saves/loads **adapter weights + config** using **safetensors + JSON**. That makes it the best fit if you want a Rust training library that already thinks in “adapter artifacts” rather than just raw tensors. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)

### 2) **Tool-calling QLoRA (LLMs): use `qlora-rs`, but treat it as an early-stage crate and plan to fork it**

`qlora-rs` is the clearest Rust-native QLoRA implementation surfaced by the research: it provides **NF4 quantization**, **double quantization**, **QLoRA training on frozen quantized weights**, **PagedAdamW**, and export paths including **GGUF** and a **native Candle quantized format**. However, it is also clearly **young** in repository maturity, so I would not consume it “as-is” in production without pinning or forking it. [\[docs.rs\]](https://docs.rs/qlora-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs)

### 3) **Image LoRA (Stable Diffusion / SDXL / related): build on Candle + `candle-lora`**

For image LoRA, the best Rust-side building block is **`candle-lora`**, not `peft-rs`, because `candle-lora` explicitly supports **Linear, Conv1d, Conv2d, and Embedding** layer conversion and **weight merge/unmerge**. That matters because diffusion LoRA training typically needs to touch **UNet** and sometimes **text encoder** components, where **Conv2d + Linear** support is essential. Candle already ships **Stable Diffusion v1.5/v2.1/SDXL/Turbo** implementations, so the most realistic Rust-native image LoRA path is to wire LoRA into those Candle model components yourself. [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/mod.rs), [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/stable_diffusion/index.html)

### 4) **Inference target: prefer Candle with merged weights first; unmerged runtime adapters second**

Because you specifically want **Candle inference**, the **lowest-risk deployment format** is usually **merged weights**: train the adapter, merge it into the base model, and then run ordinary Candle inference on the merged model. Candle loads **safetensors**, supports **quantized model formats**, and `candle-lora` explicitly supports **merge/unmerge** for LoRA. That is much less brittle than depending on runtime adapter application for every model family. [\[github.com\]](https://github.com/huggingface/candle), [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4-quantization)

***

# Best crates by use case

## A. Tool-calling LoRA / PEFT for LLMs

### **Primary pick: `peft-rs`**

Why I would choose it for tool-calling models: it is **adapter-centric**, Candle-based, supports **multiple PEFT methods**, has **save/load utilities**, and exposes a **registry** for multiple adapters and runtime switching. For a tool-calling model program, that is exactly the shape you want: fine-tune adapters, keep artifacts small, and optionally activate different adapters per tenant/task. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)

**Strengths**

*   Candle-native and Rust-native, so there is no Python runtime dependency in the adapter layer. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)
*   Saves **adapter weights in safetensors** and **config in JSON**, which aligns well with a Candle inference stack. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)
*   Supports **multi-adapter** workflows and **weight merging**, which is valuable for shipping multiple tool-calling specializations from one base model. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)
*   Gives you room to experiment beyond LoRA later (DoRA, IA³, AdaLoRA, etc.) without changing the broad architecture of your training system. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)

**Caveat**

*   It is still a **young crate** in ecosystem terms; the surfaced metadata shows **initial 1.0 releases only in January 2026** and limited public adoption signals so far. I would absolutely **pin versions and fork it** if you adopt it. [\[lib.rs\]](https://lib.rs/crates/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)

### **Secondary pick: `candle-lora`**

Use `candle-lora` instead if your LLM family is one it already supports well, because it has direct integrations for **llama, mistral, falcon, bert, stable\_lm, t5, mpt, blip, starcoder** and provides **merge/unmerge** plus a relatively ergonomic model-conversion path. If your target model is one of those families, it can be the shortest path from training to Candle inference. [\[github.com\]](https://github.com/EricLBuehler/candle-lora)

**Important compatibility note:** `candle-lora` explicitly says its **weight naming is not compatible with PEFT yet**. So if you train with `peft-rs` and infer with `candle-lora`, you should assume you will need a **name-mapping/conversion shim** unless you deploy only **merged** weights. [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[github.com\]](https://github.com/tzervas/peft-rs)

***

## B. Tool-calling QLoRA for LLMs

### **Primary pick: `qlora-rs` + your own integration layer**

This is the only clearly surfaced Rust-native QLoRA crate with the pieces you need: **NF4**, **double quantization**, **QLoraTrainer**, **PagedAdamW**, **GGUF export**, and a **native Candle quantized format**. Because Candle itself already supports **quantized model loading/inference** and GGML/GGUF-style formats, this gives you a plausible route to **train in Rust and still deploy through Candle**. [\[docs.rs\]](https://docs.rs/qlora-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[github.com\]](https://github.com/huggingface/candle), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4-quantization)

**Why it fits your workflow**

*   QLoRA’s whole point is keeping the **base model frozen in 4-bit** while training the adapter in higher precision, which is exactly what `qlora-rs` advertises. [\[docs.rs\]](https://docs.rs/qlora-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs)
*   It explicitly integrates with **`peft-rs` for adapter management**, which suggests a clean split where `qlora-rs` handles the quantized training mechanics while `peft-rs` handles adapter lifecycle and config. [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[docs.rs\]](https://docs.rs/peft-rs)
*   It can export **GGUF**, and Candle already supports **quantized LLM inference** using llama.cpp-style quantized types. [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[github.com\]](https://github.com/huggingface/candle), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4-quantization)

**But there is a real maturity warning**
The repository surfaced as extremely new: **one commit**, **no releases**, **no stars** in the GitHub snippet at crawl time. That does not make it bad, but it absolutely makes it something you should **vendor or fork** before relying on it for production fine-tuning. [\[github.com\]](https://github.com/tzervas/qlora-rs)

***

## C. Image LoRA for domain-specific generation

### **Primary pick: Candle Stable Diffusion + `candle-lora` + your own training loop**

This is the strongest Rust-native image story I found, but it is more of a **construction kit** than a turnkey product. Candle ships **Stable Diffusion v1.5/v2.1/SDXL/Turbo** model code and examples, while `candle-lora` supports the layer types you need for diffusion-style LoRA (**Conv2d**, **Linear**, **Embedding**) and also supports **merge/unmerge**. That gives you enough to build your own image LoRA training stack in Rust. [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/mod.rs), [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/stable_diffusion/index.html), [\[github.com\]](https://github.com/EricLBuehler/candle-lora)

**Why not `peft-rs` here?**
The surfaced `peft-rs` materials position it as a **general PEFT library for LLM fine-tuning**, whereas `candle-lora` explicitly supports **Conv2d** and has a broader “swap layers inside Candle models” story. For diffusion models, that matters more than the broader list of PEFT methods. [\[github.com\]](https://github.com/tzervas/peft-rs), [\[github.com\]](https://github.com/EricLBuehler/candle-lora)

**What the gap is**
I did **not** find a mature Rust equivalent of Python `diffusers`’ official **LoRA/DreamBooth image training scripts**. In Python, those recipes are documented and maintained for text-to-image LoRA and DreamBooth LoRA; in Rust, you will be assembling the training path yourself on top of Candle. That is viable, but it is not yet “turnkey Rust.” [\[huggingface.co\]](https://huggingface.co/docs/diffusers/training/lora), [\[github.com\]](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README.md), [\[github.com\]](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_qwen.md), [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion)

***

## D. Optional platform-specific pick: `metal_candle` for Apple Silicon

If you are training on Apple Silicon, `metal_candle` is worth knowing about because it is explicitly a **Metal-backed Candle-derived library** with **LoRA training utilities**, model loading, and text generation APIs. I would not make it your core cross-platform architecture, but it is a useful optimization track if your lab machines are M-series Macs. [\[docs.rs\]](https://docs.rs/metal-candle)

***

# What I would **not** make primary for your workflow

## `burn`

Burn is a strong Rust deep learning framework with a **Candle backend**, **training + inference support**, and quantization features, so it is absolutely credible as a general ML framework. However, the sources I found emphasize **generic model training/inference** and backend flexibility, not a LoRA/QLoRA adapter workflow that naturally drops into Candle model loaders. For your particular constraint—**train adapters, then serve through Candle**—it adds more integration work than starting with Candle-native adapter crates. [\[docs.rs\]](https://docs.rs/burn/latest/burn/), [\[github.com\]](https://github.com/Tracel-AI/burn), [\[lib.rs\]](https://lib.rs/crates/burn-train)

## `dfdx`

`dfdx` is a very interesting all-Rust deep learning library with CUDA support and shape-checked tensors, but the surfaced materials still describe it as **pre-alpha** and I found no direct LoRA/QLoRA adapter path aligned with Candle inference artifacts. That makes it a great research toy or foundation library, but not my first pick for your production pipeline. [\[docs.rs\]](https://docs.rs/dfdx/latest/dfdx/), [\[github.com\]](https://github.com/chelsea0x3b/dfdx)

***

# The workflow I recommend you actually implement

## Workflow 1 — **Tool-calling LLMs (LoRA)**

1.  **Choose a base model family that Candle supports and that already has the least adapter friction**, ideally **Llama, Mistral, Falcon, T5, or StarCoder-family**. Candle supports many LLMs, but `candle-lora` already has explicit transformer integrations for several of those families, which reduces your integration burden. [\[github.com\]](https://github.com/huggingface/candle), [\[github.com\]](https://github.com/EricLBuehler/candle-lora)
2.  **Train adapters with `peft-rs`** and save artifacts as **adapter safetensors + JSON config**. This gives you a cleaner adapter lifecycle than hand-rolling tensors. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)
3.  **At validation time, always produce two outputs**:
    *   **unmerged adapter artifact** for research/debugging, and
    *   **merged fp16/bf16 base model** for the simplest Candle deployment path. `peft-rs` and `candle-lora` both have merge semantics, and Candle loads standard weight files. [\[github.com\]](https://github.com/tzervas/peft-rs), [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[github.com\]](https://github.com/huggingface/candle)
4.  **Deploy Candle against the merged model first.** If you later need hot-swappable adapters, then add a runtime adapter path using either `peft-rs` inside the inference binary or a custom adapter loader around Candle. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/huggingface/candle)

## Workflow 2 — **Tool-calling LLMs (QLoRA)**

1.  **Quantize + train with `qlora-rs`**, ideally while also standardizing your adapter config/layout to be compatible with your `peft-rs` conventions. `qlora-rs` already advertises integration with `peft-rs`. [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[docs.rs\]](https://docs.rs/peft-rs)
2.  **Do not assume direct unmerged Candle QLoRA inference on day one.** Instead, make your first production path one of these:
    *   **merge adapter into base, then export/deploy**, or
    *   **export GGUF** and use Candle’s quantized model path where that fits the model family. [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[github.com\]](https://github.com/huggingface/candle), [\[deepwiki.com\]](https://deepwiki.com/huggingface/candle/4-quantization)
3.  **Pin/fork `qlora-rs` early** and write tests around your exact model family and export path, because its public maturity is currently low. [\[github.com\]](https://github.com/tzervas/qlora-rs)

## Workflow 3 — **Image generation LoRA**

1.  **Start from Candle’s Stable Diffusion implementation** for the exact image family you want to support first—**v1.5**, **v2.1**, or **SDXL** are the obvious candidates because Candle already has model code/examples for them. [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/mod.rs), [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/stable_diffusion/index.html)
2.  **Inject LoRA using `candle-lora`-style conversion for UNet and optionally text encoder modules.** The reason this is the right crate is that it already supports **Conv2d/Linear/Embedding** and **merge/unmerge**. [\[github.com\]](https://github.com/EricLBuehler/candle-lora)
3.  **Save adapter weights in safetensors**, but for actual deployment **prefer merged UNet/text-encoder weights** so your Candle image inference path remains plain Candle Stable Diffusion. [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[github.com\]](https://github.com/huggingface/candle)
4.  **Expect to own the training loop** (noise schedule, latent prep, text conditioning, optimizer, checkpointing), because I did not find a mature Rust diffusion-LoRA training stack equivalent to `diffusers`’ maintained scripts. [\[huggingface.co\]](https://huggingface.co/docs/diffusers/training/lora), [\[github.com\]](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README.md), [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion)

***

# The artifact contract I would standardize on

I strongly recommend that your whole system revolve around a **canonical adapter artifact contract**:

### For LoRA

*   `adapter_config.json`
*   `adapter_weights.safetensors`
*   `base_model_id` / commit hash metadata
*   `target_modules` / module mapping manifest
*   `merge_recipe.json` for deterministic merge behavior [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/peft-rs)

### For QLoRA

*   everything above, plus
*   `quantization_config.json` (NF4 block size, double quantization flags, compute dtype, etc.)
*   optional exported **GGUF** or **native Candle quantized** artifact for deployment experiments. [\[docs.rs\]](https://docs.rs/qlora-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs)

Why this matters: Candle can consume standard model weights and quantized formats, while `peft-rs`, `candle-lora`, and `qlora-rs` all already think in terms of **separate adapters + configs + merge/export steps**. Standardizing your own manifest will save you from naming mismatches later. [\[github.com\]](https://github.com/huggingface/candle), [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs)

***

# My final recommendation, in one sentence

If I were building your stack, I would use **`peft-rs` for LLM LoRA**, **`qlora-rs` (forked) for LLM QLoRA**, **`candle-lora` + Candle Stable Diffusion for image LoRA**, and I would make **merged Candle-ready weights the default deployment artifact**, with runtime adapters only as a second-phase feature. [\[docs.rs\]](https://docs.rs/peft-rs), [\[github.com\]](https://github.com/tzervas/qlora-rs), [\[github.com\]](https://github.com/EricLBuehler/candle-lora), [\[github.com\]](https://github.com/huggingface/candle/tree/main/candle-examples/examples/stable-diffusion), [\[github.com\]](https://github.com/huggingface/candle)

***

If you want, I can take this one step further and give you either:

1.  a **concrete reference architecture** (repos/crates/services/files) for this workflow, or
2.  a **decision matrix by model family** (e.g., Mistral vs Llama vs Qwen vs SDXL) showing exactly which crate path I’d use for each.


## The Architecture

Absolutely — below is a **reference architecture** I’d recommend for a **Rust-first LoRA / QLoRA workflow** where:

*   **training is in Rust whenever practical**, [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle)
*   **inference is done with Candle**, [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft)
*   **tool-calling models** use **LoRA / QLoRA**, [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle)
*   **image models** use **LoRA** on top of Candle’s Stable Diffusion-family implementations, [\[danielkliewer.com\]](https://www.danielkliewer.com/blog/2025-03-11-integrating-rust-burn-framework-for-ai), [\[lib.rs\]](https://lib.rs/crates/burn-train), [\[docs.rs\]](https://docs.rs/dfdx/latest/dfdx/), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)
*   and your system keeps **adapter artifacts, merged artifacts, and quantized artifacts** cleanly separated. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)

I’ll show this in layers:

1.  **Overall platform architecture**
2.  **Training architecture**
3.  **Artifact flow**
4.  **Inference architecture**
5.  **Recommended repo / crate layout**
6.  **Suggested workflow phases**

***

# 1) Design goals for the reference architecture

I’d optimize the architecture around these principles:

*   **Candle is the runtime boundary for inference**, because Candle already supports model loading, training primitives, safetensors, and quantized model support across many LLM and image model families. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft), [\[github.com\]](https://github.com/tzervas/peft-rs/tree/main/src/adapters)
*   **Adapters are first-class artifacts**, because `peft-rs` already saves adapter configs + weights, and `candle-lora` supports merge/unmerge flows, which makes adapters practical units of versioning and deployment. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)
*   **Merged models are the default production path**, because both `peft-rs` and `candle-lora` expose merge concepts, and merged weights simplify Candle inference substantially compared with dynamic adapter composition. [\[rustrepo.com\]](https://rustrepo.com/tag/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)
*   **QLoRA is treated as a specialized training/export lane**, because `qlora-rs` provides NF4, double quantization, training, and export, but it is much less mature than Candle or Candle-adjacent crates. [\[lib.rs\]](https://lib.rs/crates/peft-rs)
*   **Image LoRA is separated from LLM LoRA**, because diffusion LoRA has different target modules and lifecycle needs, and `candle-lora`’s support for `Conv2d`, `Linear`, and `Embedding` makes it a better fit than a generic LLM-only PEFT layer. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[lib.rs\]](https://lib.rs/crates/burn-train), [\[docs.rs\]](https://docs.rs/dfdx/latest/dfdx/)

***

# 2) High-level reference architecture



This architecture is grounded in the fact that Candle supports **training + inference**, loads **standard model weights**, and supports **quantized types**, while `peft-rs`, `candle-lora`, and `qlora-rs` all revolve around explicit adapter/export artifacts. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft), [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

***

# 3) Recommended component architecture

## A. Data and dataset layer

You said you’ll own **data and labeling**, so I would keep that independent from the model frameworks and output a **canonical manifest format** that every trainer consumes. That prevents lock-in to one trainer implementation and makes it easy to reuse the same samples for **LoRA**, **QLoRA**, and **image LoRA** experiments. This is a design recommendation, but it aligns well with the fact that the Rust-side crates are still more modular/foundation-style than end-to-end trainer ecosystems. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[danielkliewer.com\]](https://www.danielkliewer.com/blog/2025-03-11-integrating-rust-burn-framework-for-ai)

## B. Training layer

I would split training into **three separate trainer binaries/services**:

*   **`trainer-llm-lora`** → built on **`peft-rs`** for tool-calling and structured-output LLM adaptation. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   **`trainer-llm-qlora`** → built on **`qlora-rs`** plus `peft-rs` conventions for artifact management. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle)
*   **`trainer-image-lora`** → built on **Candle Stable Diffusion modules** and **`candle-lora`** for UNet/text-encoder adaptation. [\[danielkliewer.com\]](https://www.danielkliewer.com/blog/2025-03-11-integrating-rust-burn-framework-for-ai), [\[lib.rs\]](https://lib.rs/crates/burn-train), [\[docs.rs\]](https://docs.rs/dfdx/latest/dfdx/), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

## C. Artifact layer

Your platform should distinguish at least **three deployable artifact types**:

1.  **Adapter package** — config + adapter weights only. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
2.  **Merged model package** — base weights with adapter already merged. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
3.  **Quantized deployment package** — QLoRA-derived or post-merged quantized artifact, potentially including GGUF. [\[github.com\]](https://github.com/tzervas/peft-rs/tree/main/src/adapters)

## D. Inference layer

The safest production path is:

*   **load merged model in Candle**, then
*   serve task-specific inference with your normal Candle runtime. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

You can later add **runtime adapter loading** as a second phase if you want hot-swappable adapters, because both `peft-rs` and `candle-lora` expose patterns for runtime adapter management / merging. [\[rustrepo.com\]](https://rustrepo.com/tag/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

***

# 4) Detailed training architecture



This split mirrors the actual crate strengths:

*   `peft-rs` is the cleanest **LLM adapter** abstraction I found in Rust. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   `qlora-rs` explicitly provides **NF4**, **double quantization**, **QLoraTrainer**, and **PagedAdamW**. [\[lib.rs\]](https://lib.rs/crates/peft-rs)
*   `candle-lora` supports **Conv2d / Linear / Embedding** replacement and **weight merging**, which is what makes image LoRA feasible in Candle. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

***

# 5) Artifact contract architecture

I’d strongly recommend defining your own **internal artifact contract** even if the training crates already have their own save/load APIs.



Why this helps:

*   `peft-rs` already exposes **save/load adapter config** and **save/load adapter weights** utilities. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   `candle-lora` supports adapter tensor retrieval and weight merge/unmerge behavior. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)
*   `qlora-rs` adds **quantization-specific state** that you want recorded separately from plain LoRA metadata. [\[lib.rs\]](https://lib.rs/crates/peft-rs)

### Suggested artifact fields

I recommend every artifact bundle include:

*   `base_model_id` and exact revision/hash, because Candle supports many families and you need strict compatibility tracking. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft)
*   `adapter_method` (`lora`, `qlora`, `adalora`, etc.), because `peft-rs` supports many methods and you do not want ambiguity in deployment. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   `target_modules`, because adapter placement determines whether merge/runtime application works correctly. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   `quantization_config`, when relevant, because QLoRA export and reload depend on NF4/double-quant choices. [\[lib.rs\]](https://lib.rs/crates/peft-rs)

***

# 6) Inference architecture

## Production inference path (recommended)



This is the path I’d recommend first because Candle already supports **model loading**, **quantized models**, and broad model-family inference, and merged weights remove most adapter-runtime complexity. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft), [\[github.com\]](https://github.com/tzervas/peft-rs/tree/main/src/adapters)

## Optional runtime adapter path



This is useful when you want **hot-swappable tenants/tasks**, but it is phase-two architecture, not phase-one. `peft-rs` provides multi-adapter ideas, and `candle-lora` supports merge/unmerge, but merged weights are still simpler operationally. [\[rustrepo.com\]](https://rustrepo.com/tag/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

***

# 7) Image-specific reference architecture



This design follows directly from the fact that Candle already includes **Stable Diffusion-family model code**, while `candle-lora` supports the **Conv2d / Linear / Embedding** conversions that diffusion LoRA needs. [\[danielkliewer.com\]](https://www.danielkliewer.com/blog/2025-03-11-integrating-rust-burn-framework-for-ai), [\[lib.rs\]](https://lib.rs/crates/burn-train), [\[docs.rs\]](https://docs.rs/dfdx/latest/dfdx/), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)

***

# 8) Suggested repo / crate structure

I’d structure the codebase like this:

```text
workspace/
  crates/
    data-contracts/
    artifact-contracts/
    common-tokenization/
    common-eval/
    trainer-llm-lora/
    trainer-llm-qlora/
    trainer-image-lora/
    exporter-merge/
    exporter-quantize/
    inference-candle-core/
    inference-candle-llm/
    inference-candle-image/
  services/
    eval-runner/
    registry-sync/
    inference-api/
```

### Why this split works

*   It isolates **artifact compatibility logic** from trainer logic, which is important because `peft-rs`, `candle-lora`, and `qlora-rs` are not guaranteed to share identical naming conventions or serialization assumptions. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)
*   It lets you keep **Candle inference** stable while trainer implementations evolve independently. Candle is the strongest common serving substrate across the options we researched. [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft), [\[github.com\]](https://github.com/tzervas/peft-rs/tree/main/src/adapters)

***

# 9) My opinionated implementation order

I would implement this in the following sequence:

1.  **LLM LoRA end-to-end first** using **`peft-rs` → merged Candle inference**. That gives you the shortest path to a stable production loop. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[rustrepo.com\]](https://rustrepo.com/tag/candle), [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft)
2.  **Image LoRA second** using **Candle Stable Diffusion + `candle-lora`**, because it is feasible but more custom. [\[danielkliewer.com\]](https://www.danielkliewer.com/blog/2025-03-11-integrating-rust-burn-framework-for-ai), [\[lib.rs\]](https://lib.rs/crates/burn-train), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html)
3.  **QLoRA third** using a **forked `qlora-rs`**, because it is promising but clearly less mature and should be treated as a controlled engineering investment. [\[lib.rs\]](https://lib.rs/crates/peft-rs)
4.  **Runtime adapter loading fourth**, only after merged-model deployment is boring and reliable. [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[rustrepo.com\]](https://rustrepo.com/tag/candle)

***

# 10) Final summary

If I translate all of this into one practical statement:

> Build a **three-lane training platform** (**LLM LoRA**, **LLM QLoRA**, **Image LoRA**) around a **shared artifact contract**, then make **merged Candle-ready models** your standard deployment target, with **adapter-only** and **quantized** packages as secondary outputs for experimentation and future runtime flexibility. [\[sourcegraph.com\]](https://sourcegraph.com/github.com/huggingface/candle), [\[docs.rs\]](https://docs.rs/ruvllm/latest/ruvllm/backends/index.html), [\[apxml.com\]](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft)

If you want, I can next turn this into either:

1.  a **concrete workspace skeleton** with Cargo crates and responsibilities, or
2.  a **more detailed sequence diagram** for **tool-calling LoRA/QLoRA training + evaluation + Candle serving**.

