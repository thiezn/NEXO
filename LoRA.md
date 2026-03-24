# LoRA explanation

## My standard approach to modelling LoRA datasets.

What token/trigger word do I use to trigger LoRA behaviour. I want to have re-usable tokens in all my training sets so I can create a better fly wheel for gathering proper datasets for labelling.


# GPT5.4 excellent explanation

Absolutely — **LoRA** is one of the most useful ideas in modern model fine-tuning because it lets you adapt a large model **without retraining all of its weights**.

***

# What is LoRA?

**LoRA** stands for **Low-Rank Adaptation**.

The core idea is:

> Instead of changing the full weight matrix of a large neural network during fine-tuning, you keep the original weights frozen and learn a **small low-rank update** that gets added on top.

## Intuition

A neural network layer often has a large weight matrix:

$$
W \in \mathbb{R}^{d \times k}
$$

If you fully fine-tune the model, you update **every value in $$W$$**.

With LoRA, you leave $$W$$ unchanged and learn a small update:

$$
\Delta W = B A
$$

where:

*   $$A \in \mathbb{R}^{r \times k}$$
*   $$B \in \mathbb{R}^{d \times r}$$
*   $$r$$ is a **small rank** (for example 4, 8, 16, 32)

So instead of learning $$d \times k$$ parameters, you learn only:

$$
r \times k + d \times r
$$

If $$r$$ is much smaller than $$d$$ and $$k$$, this is **far fewer parameters**.

The effective weight used at inference becomes:

$$
W' = W + \alpha \cdot B A
$$

where **α** is a scaling factor.

***

# Why LoRA is useful

LoRA became popular because it solves several practical problems:

## 1. Much cheaper than full fine-tuning

You only train a tiny fraction of the parameters.

## 2. Smaller storage

Instead of saving a full copy of a multi-GB model, you save a relatively small adapter file.

## 3. Easy to swap

You can keep one base model and load different LoRAs for different tasks.

## 4. Less infrastructure cost

Training and serving are easier because most of the model remains frozen.

## 5. Composable

In many systems, multiple LoRAs can be merged or stacked for different behaviors.

***

# A simple mental model

Think of the base model as a **generalist**.

A LoRA is like a **lightweight specialization patch**:

*   Base model: “knows language/images broadly”
*   LoRA: “nudges it toward a task/style/behavior”

It doesn’t usually teach the model everything from scratch. It **biases** the model in a useful direction.

***

# How LoRA works in practice

LoRA is usually applied to specific layers, especially in **attention** and sometimes **MLP** layers.

For transformers, common targets are projections like:

*   **Q** (query)
*   **K** (key)
*   **V** (value)
*   **O** (output)

In some setups, only a subset is adapted (for example Q and V). In others, more layers are included for stronger adaptation.

***

# LoRA in image generation models

In image generation, “LoRA” often refers to a small adapter trained on top of a base model like a diffusion model.

This is extremely common in ecosystems around:

*   Stable Diffusion
*   SDXL
*   FLUX-style derivative ecosystems
*   custom anime/photoreal/style checkpoints

## What it does in image models

A LoRA can teach the base image model to better represent:

*   a **specific visual style**
*   a **particular character**
*   a **person’s face**
*   a **specific object**
*   a **clothing concept**
*   a **camera/lighting aesthetic**
*   a **pose/style combination**

So instead of training a whole new checkpoint, you train a small adapter that says:

> “When the prompt contains certain tokens or conditions, shift the model’s internal behavior in this direction.”

***

## Example

Suppose you have a base image model that can generate portraits.

You want it to generate images in a very specific style:

*   watercolor anime
*   cyberpunk product photography
*   your company mascot
*   a custom game character

Rather than fully fine-tuning the model, you train a LoRA on a smaller dataset of images representing that concept.

Then during inference you use the base model + LoRA.

The LoRA alters how the model denoises and interprets prompt features, making the output more likely to match that learned concept.

***

## Why LoRA is so popular for image generation

### 1. Small file size

A LoRA file can be dramatically smaller than a full model checkpoint.

### 2. Easy distribution

Creators can share LoRAs rather than entire models.

### 3. Easy combination

People often combine:

*   a base model
*   one style LoRA
*   one character LoRA
*   one detail enhancer LoRA

### 4. Faster experimentation

You can iterate on many artistic concepts cheaply.

### 5. Lower data requirement than full retraining

For narrow concepts, LoRA works surprisingly well even with relatively modest datasets (though quality still depends heavily on data quality).

***

## How LoRA is injected into image generation models

In diffusion models, LoRA is usually applied to parts of the **U-Net** and/or the **text encoder**.

### U-Net LoRA

This affects how latent images are denoised and refined.

### Text encoder LoRA

This affects how prompt tokens get interpreted.

Training either or both changes how strongly the prompt maps to visual outcomes.

***

## What users experience in image generation

From the user point of view, a LoRA is often controlled by:

*   a **trigger word** or phrase in the prompt
*   a **weight**, e.g. `<lora:my_style:0.7>`

That weight determines how strongly the LoRA influences the output.

### Too low:

*   effect is weak
*   concept may not show up clearly

### Too high:

*   image can become distorted
*   style may overpower the prompt
*   artifacts can appear

So LoRA strength is usually tuned experimentally.

***

## Common image-generation LoRA use cases

### Style LoRA

Teaches a specific artistic style.

### Character LoRA

Teaches a recognizable person/character appearance.

### Object LoRA

Teaches a specific product, logo-like object, or design motif.

### Pose/composition LoRA

Biases the model toward certain framing or body configurations.

### Realism/detail LoRA

Enhances sharpness, texture, skin detail, lighting aesthetics, etc.

***

## Limitations in image models

LoRAs are powerful, but not magic.

### 1. Overfitting

If trained on too few or too similar images, the model memorizes instead of generalizing.

### 2. Prompt brittleness

The LoRA may work well only with certain trigger words or prompt styles.

### 3. Conflicts with other LoRAs

Multiple LoRAs may fight each other.

### 4. Limited conceptual depth

A small adapter is great for narrow specialization, but not always enough for large capability shifts.

### 5. Base-model dependence

A LoRA trained on one base model often won’t work well on another unless architectures and training assumptions align.

***

# LoRA in tool-calling models

Now to the second part: **tool calling**.

A tool-calling model is a language model trained to do things like:

*   decide whether to call a tool/API
*   choose the correct tool
*   produce structured arguments (often JSON)
*   return control to the application

Examples:

*   weather API calls
*   search tools
*   database lookups
*   calendar creation
*   code execution
*   retrieval calls

***

## What LoRA does here

In a tool-calling model, LoRA can fine-tune a base LLM to become better at:

1.  **Recognizing when a tool is needed**
2.  **Selecting the right tool**
3.  **Formatting the tool call correctly**
4.  **Extracting arguments from the user’s request**
5.  **Following a schema**
6.  **Reducing hallucinated tool calls**

So instead of changing the whole language model, you apply a compact adaptation that nudges the model toward structured behavior.

***

## Example

Suppose you start with a general instruction model.

User says:

> “Book a meeting with Sarah tomorrow at 3 PM and add a Teams link.”

A generic model may answer in plain text.

A tool-call LoRA can teach it to output something more like:

```json
{
  "tool": "create_calendar_event",
  "arguments": {
    "title": "Meeting with Sarah",
    "datetime": "2026-03-25T15:00:00",
    "add_teams_link": true
  }
}
```

This is exactly the kind of behavioral specialization LoRA is good at.

***

# Why LoRA is attractive for tool calling

## 1. Efficient specialization

You can take a strong general-purpose model and cheaply specialize it for:

*   function calling
*   JSON schema compliance
*   enterprise workflows
*   domain-specific APIs

## 2. Multiple adapters for multiple environments

A company might keep one base model and attach different LoRAs for:

*   CRM tools
*   cloud ops tools
*   ITSM workflows
*   finance tools
*   internal search/retrieval systems

## 3. Safer iteration

You can improve tool-use behavior without retraining the full model.

## 4. Domain adaptation

If your API names, parameter conventions, or business objects are unique, LoRA can teach the model that vocabulary.

***

# What is actually learned in tool-calling LoRAs?

A tool-calling LoRA often teaches patterns like:

## 1. Tool selection policy

When should the model answer directly vs. invoke a tool?

## 2. Argument extraction

How to map natural language into structured parameters.

Example:

*   “next Friday morning” → date range / timestamp
*   “the Berlin office” → office\_id or location field

## 3. Schema obedience

The model learns to produce required fields, enum values, nesting, and valid JSON.

## 4. Tool sequencing

In more advanced setups:

*   search first
*   then retrieve details
*   then call action tool

## 5. Error-aware behavior

Some fine-tuning datasets teach the model to:

*   ask for missing fields
*   retry with corrected arguments
*   interpret tool results before answering

***

# LoRA vs prompt engineering for tool calling

Prompting alone can get you surprisingly far, especially with strong frontier models.

But LoRA can help when you need **consistency**.

## Prompt engineering is good for:

*   quick iteration
*   small number of tools
*   strong base model
*   low-stakes workflows

## LoRA is better when:

*   you need reliable structured output
*   your schemas are complicated
*   the tool vocabulary is domain-specific
*   you have enough training data
*   you want lower latency / less giant prompting
*   you need consistent behavior across many requests

In other words:

> Prompting tells the model what you want now.  
> LoRA changes how the model tends to behave.

***

# Typical training data for a tool-calling LoRA

You would train on examples like:

*   user query
*   available tools / schemas
*   expected tool call
*   maybe intermediate reasoning traces (depending on setup)
*   maybe tool responses and final answer

For example:

### Input

“Restart the failing VM in resource group prod-eu and tell me whether it comes back healthy.”

### Target behavior

1.  call `get_vm_status`
2.  if unhealthy, call `restart_vm`
3.  call `get_vm_status` again
4.  summarize result

This kind of supervised data teaches orchestration patterns.

***

# LoRA in tool calling: where it is applied

Technically, the LoRA is still usually attached to transformer layers in the language model — often attention projections and sometimes MLP blocks.

The difference is not *where* it is inserted, but *what data it is trained on*.

So:

*   **Image LoRA** → data teaches visual concepts/styles
*   **Tool-calling LoRA** → data teaches structured decision-making and API usage

Same mechanism, different specialization objective.

***

# LoRA vs QLoRA

You may also hear **QLoRA**.

This is related but different.

## LoRA

Adds low-rank trainable adapters to a frozen base model.

## QLoRA

Does the same thing, **but** the base model is quantized (often 4-bit) during fine-tuning to reduce memory usage even more.

This is especially popular for LLM fine-tuning because it makes training large models much cheaper.

For tool-calling adaptation, QLoRA is often a very practical choice.

***

# Strengths and weaknesses of LoRA overall

## Strengths

*   parameter efficient
*   cheaper to train
*   small artifacts to store/share
*   easy to swap/merge
*   great for narrow specialization
*   excellent for experimentation

## Weaknesses

*   not always enough for deep capability shifts
*   can overfit
*   can be sensitive to rank / target modules / scaling
*   may interfere with other adapters
*   performance ceiling can be below full fine-tuning on some tasks

***

# A concise comparison

## In image generation

LoRA is used to teach:

*   styles
*   characters
*   objects
*   aesthetics

It modifies how the diffusion model interprets prompts and denoises images.

## In tool-calling models

LoRA is used to teach:

*   when to call tools
*   which tool to use
*   how to produce arguments
*   how to follow structured schemas

It modifies how the language model performs structured reasoning and action formatting.

***

# One-sentence summary

> **LoRA is a lightweight fine-tuning method that learns a small low-rank “patch” for a frozen base model, making it ideal for cheaply specializing image models toward styles/concepts and language models toward reliable tool-calling behavior.**


## GPT5.4 QLoRA deep dive

Absolutely — **QLoRA** is essentially **LoRA made much more memory-efficient**, which is why it became a big deal for fine-tuning large language models on relatively modest hardware.

***

# What is QLoRA?

**QLoRA** stands for **Quantized Low-Rank Adaptation**.

It combines two ideas:

1.  **LoRA**: fine-tune a model by learning small low-rank adapter matrices instead of updating all model weights.
2.  **Quantization**: store the frozen base model in a low-precision format (commonly **4-bit**) instead of full precision like 16-bit or 32-bit.

So instead of:

*   loading the full base model in high precision, and
*   fine-tuning all or most of it,

QLoRA does this:

*   **quantize the base model**
*   **freeze it**
*   **train only the LoRA adapters**

That drastically reduces memory usage while preserving surprisingly strong performance.

***

# The core idea

In normal LoRA:

*   the base model stays frozen
*   small trainable adapter weights are added
*   base model is often still loaded in 16-bit precision (or similar)

In **QLoRA**:

*   the base model is loaded in **4-bit quantized form**
*   the model is still frozen
*   training happens only in the LoRA layers
*   the quantized model is used during forward/backward passes in a careful way that preserves training quality

So the key innovation is:

> **Use a compressed base model for memory savings, while still training high-quality LoRA adapters on top.**

***

# Why this matters

Large models are expensive to fine-tune because memory gets consumed by:

*   model weights
*   gradients
*   optimizer states
*   activations

Even if LoRA already reduces trainable parameters, the **base model itself still takes a lot of memory**.

QLoRA attacks that remaining problem by shrinking the frozen model footprint.

This makes it possible to fine-tune models that would otherwise require much more expensive GPUs.

***

# How QLoRA works conceptually

Here’s the simplified pipeline:

## 1. Start with a pretrained model

For example, a language model with billions of parameters.

## 2. Quantize the frozen base model

Store the base weights in a compact format, typically **4-bit**.

## 3. Add LoRA adapters

Insert trainable low-rank matrices into selected layers (usually attention projections, sometimes more).

## 4. Train only the adapters

The base weights do not change; only the LoRA parameters update.

## 5. Use the adapted model

At inference time, the output reflects:

*   the original pretrained model
*   plus the learned LoRA updates

***

# Why QLoRA is different from “just quantization”

It’s important not to confuse **quantizing a model for inference** with **QLoRA for fine-tuning**.

## Plain quantization

This is mostly about making inference cheaper:

*   less VRAM
*   faster loading
*   lower hardware requirements

## QLoRA

This is about **fine-tuning efficiently**

*   quantized frozen model
*   trainable LoRA adapters
*   memory-efficient training recipe

So QLoRA is not just “run a model in 4-bit.”  
It is a **specific fine-tuning strategy**.

***

# Main advantages of QLoRA

## 1. Dramatically lower memory usage

This is the biggest advantage.

A huge model in 16-bit precision can consume a lot of GPU memory just to load.  
Reducing the frozen base model to 4-bit can cut that footprint dramatically.

That means you can fine-tune much larger models on hardware that would otherwise be insufficient.

### Why this is valuable

*   fewer / cheaper GPUs needed
*   easier experimentation
*   more practical for individuals and smaller teams
*   lower cloud cost

***

## 2. Near-full-fine-tuning quality at much lower cost

A major reason QLoRA became popular is that it often gives **performance close to much more expensive fine-tuning approaches**, especially for narrow adaptation tasks.

In many real workloads, QLoRA is “good enough” or even very strong relative to its cost.

### In practice

This makes it a great choice for:

*   instruction tuning
*   domain adaptation
*   chat formatting alignment
*   tool-calling specialization
*   enterprise workflow adaptation

***

## 3. Much cheaper experimentation

If you are trying:

*   different datasets
*   different prompts/templates
*   different adapter ranks
*   different training objectives

QLoRA lowers the barrier to iteration.

Instead of treating each fine-tuning run as a big infrastructure decision, you can test ideas faster and more cheaply.

***

## 4. Keeps the base model intact

Like LoRA, QLoRA does **not overwrite the original pretrained weights**.

That means:

*   you preserve the base model
*   you save only the adapter
*   you can swap adapters for different tasks

This is operationally very convenient.

For example, one base model could have separate QLoRA-trained adapters for:

*   customer support
*   internal IT ticketing
*   cloud operations
*   finance report extraction
*   tool calling

***

## 5. Small adapter artifacts

You do not need to save an entire fine-tuned model checkpoint every time.

Instead, you often save just:

*   the LoRA adapter weights
*   some config metadata

This makes deployment and sharing easier.

***

## 6. Good fit for domain-specific adaptation

QLoRA is especially good when you want to “nudge” a model into a specialized behavior rather than teach it an entirely new capability from scratch.

Examples:

*   make a general LLM better at Azure troubleshooting
*   make a chatbot follow a company’s answer style
*   improve extraction of fields from invoices
*   specialize a model for function calling / structured JSON output

For these kinds of tasks, QLoRA often hits a very strong cost/performance tradeoff.

***

# Why QLoRA is especially popular for LLMs

LLMs are big enough that memory is often the primary bottleneck.

For many practitioners, the question is not:

> “What is the theoretically best fine-tuning approach?”

It is:

> “What can I actually train on the hardware I have?”

QLoRA gives a strong answer to that.

It made it feasible to adapt models with:

*   fewer GPUs
*   smaller GPUs
*   lower VRAM budgets
*   consumer or prosumer setups in some cases

That practical accessibility is a huge part of its success.

***

# What does “4-bit” really mean here?

Normally, model weights might be stored in:

*   **FP32** (32-bit float)
*   **FP16** / **BF16** (16-bit)

QLoRA stores the frozen base model in a **4-bit representation**.

That means each weight uses much less memory.

Of course, this introduces approximation error.  
The trick is that:

*   the base model is frozen
*   the quantization is designed carefully
*   the trainable LoRA adapters compensate for task-specific adaptation

So despite compressing the base model heavily, performance can remain surprisingly good.

***

# Additional technical ideas often associated with QLoRA

When people talk about QLoRA, they often also mean the practical recipe around it, including techniques like:

## NF4 quantization

A 4-bit quantization format designed to better preserve weight distributions.

## Double quantization

Compress quantization constants themselves to save additional memory.

## Paged optimizers

Help manage memory spikes during training.

You don’t need to understand these to get the core concept, but they are part of why QLoRA works well in practice.

***

# QLoRA vs LoRA

## LoRA

*   base model usually loaded in higher precision
*   train only small adapter weights
*   lower training cost than full fine-tuning

## QLoRA

*   base model loaded in quantized low precision (often 4-bit)
*   still train only small adapter weights
*   even lower memory usage than standard LoRA

So a simple summary is:

> **LoRA reduces the number of trainable parameters.**  
> **QLoRA reduces trainable parameters *and* the memory footprint of the frozen base model.**

***

# QLoRA vs full fine-tuning

## Full fine-tuning

*   update all or most model parameters
*   highest training cost
*   large memory usage
*   large checkpoint files
*   can give maximum flexibility

## QLoRA

*   update only adapters
*   much cheaper
*   much smaller memory footprint
*   smaller artifacts
*   often comparable for many downstream tasks

If your goal is efficient specialization, QLoRA is usually much more practical.

If your goal is a very large behavior shift and you have substantial compute, full fine-tuning may still outperform it in some scenarios.

***

# Where QLoRA shines

QLoRA is especially strong when:

*   you have limited GPU memory
*   you need a cost-effective fine-tuning approach
*   your task is a specialization of existing model capabilities
*   you want multiple adapters for different domains
*   you need a reproducible and relatively lightweight training workflow

Typical use cases:

*   chat assistant tuning
*   instruction following
*   internal enterprise copilots
*   code/task formatting
*   retrieval-augmented answer style adaptation
*   tool-calling behavior
*   structured extraction

***

# Limitations of QLoRA

QLoRA is powerful, but it is not magic.

## 1. Not always equal to full fine-tuning

For some tasks, especially ones requiring major representational changes, full fine-tuning may still do better.

## 2. Quantization can introduce quality loss

Even though the results are often strong, aggressive quantization is still an approximation.

## 3. Still requires careful training choices

You still need to tune:

*   LoRA rank
*   target modules
*   learning rate
*   batch size
*   dataset quality
*   prompt formatting

## 4. Base-model dependence

Your results depend heavily on the quality and suitability of the starting model.

## 5. Best for adaptation, not miracles

QLoRA works best when the model already has the underlying capability and you want to refine or steer it.

***

# A practical mental model

Think of it like this:

*   **Full fine-tuning** = renovate the entire building
*   **LoRA** = keep the building, add modular extensions
*   **QLoRA** = keep the building compressed for storage efficiency, and add modular extensions on top

You get most of the specialization benefit without paying the full cost of rebuilding everything.

***

# One-sentence summary

> **QLoRA is a memory-efficient fine-tuning method that loads the frozen base model in low-bit quantized form (typically 4-bit) and trains only small LoRA adapters, enabling strong model adaptation at a fraction of the hardware and storage cost of full fine-tuning.**

***

# Short “advantages only” recap

The main advantages of QLoRA are:

*   **much lower VRAM usage**
*   **cheaper fine-tuning**
*   **can adapt larger models on smaller hardware**
*   **small adapter files**
*   **keeps the base model unchanged**
*   **good performance for many practical tasks**
*   **great for experimenting and maintaining multiple task-specific adapters**


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

