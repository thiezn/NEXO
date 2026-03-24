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
