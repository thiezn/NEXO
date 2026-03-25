## Question

I am building my own local inference system running on Macbook M4 Max 36GB and use rust and candle to load a model, eg. gemma3, in memory and run inference through it.

I'm not sure what KV cache I'm using there and what candle, or my implentation exactly does with this. Will it likely keep growing the KV cache until memory runs out?

I want to be able to control this a bit with a separate process running in my inference code. Monitor how large the KV cache is, or how long ago a prompt came in and then clear the cache, or summarize the full generated tokens into a short compaction, clear the cache, warm up the cache again with the generated summary, ready for the next prompt

## Answer

Yes — if you are using Candle in the “normal” way for autoregressive generation, then **you almost certainly are using a KV cache**, and in Candle it is often part of the model’s mutable inference state rather than a separate cache object you can inspect directly. In Candle’s non-quantized Gemma API, the model exposes `forward(&mut self, input_ids, seqlen_offset)` and `clear_kv_cache(&mut self)`, which is a strong signal that the model instance itself holds mutable KV-cache state. [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html), [\[github.com\]](https://github.com/huggingface/candle/issues/1918)

For **upstream quantized models** in `candle-transformers`, there is also evidence that the KV cache is embedded **inside** the model object: the `candle_pipelines_models` crate explicitly calls out this design as a problem, describing upstream quantized models as using `forward(&mut self, input: &Tensor, offset: usize)` with **internal KV cache**, and saying their fix is to externalize the cache so one model can be shared across conversations while each conversation gets its own cache. [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

***

# What this likely means for your Gemma 3 + Candle setup

## Short version

If you keep calling `forward(..., offset)` on the **same model instance** and never clear/reset it, the model’s cache will generally keep accumulating past tokens for that conversation/request. A KV cache stores the keys and values for prior tokens so they can be reused on later decode steps, and in dynamic-cache style systems it grows as generation progresses. [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md), [\[github.com\]](https://github.com/huggingface/candle/issues/1918)

I would **not assume Candle is automatically doing smart eviction** for you unless you have explicitly implemented it or the specific model implementation documents a sliding-window cache. Hugging Face’s general transformer docs note that dynamic KV caches grow during generation, and only stop growing automatically for architectures/layers that explicitly use sliding-window or chunked attention. [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md)

For Candle specifically, the quantized Gemma 3 example keeps a single mutable `model`, repeatedly calls `model.forward(&input, pos_or_offset)`, and also manually truncates the prompt to a chosen `max_seq_len = 8192` before generation. That manual truncation is a strong hint that **you are expected to manage sequence growth at the application layer**, rather than relying on automatic cache eviction. [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

***

# Will it keep growing until memory runs out?

## In practice: **it can**, unless you impose a limit

A KV cache for autoregressive decoding grows with sequence length. One useful rule of thumb is that the cache size per sequence scales roughly like:

$$
2 \times L \times s \times h_{kv} \times d_{head} \times \text{bytes\_per\_element}
$$

where $$L$$ is layers, $$s$$ is cached sequence length, $$h_{kv}$$ is number of KV heads, and $$d_{head}$$ is head dimension. That means longer chats and more active sessions linearly increase KV-memory pressure. [\[deepwiki.com\]](https://deepwiki.com/huggingface/transformers/4.1-generation-configuration-and-strategies), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

So if your implementation keeps one long-lived model/session and continuously appends more dialogue, then yes: **the effective cache footprint will keep increasing with cached tokens** until one of these happens:

1.  you hit a model/application max sequence limit,
2.  you manually truncate/reset, or
3.  you run out of available memory / hit allocation failure. [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

On Apple Silicon, because memory is unified, this can feel even more “system-wide” than on a discrete GPU: model weights + activations + KV cache all compete within the same machine memory budget. Candle explicitly supports Metal / Mac acceleration, but that does **not** imply automatic KV lifecycle management. [\[huggingfac....github.io\]](https://huggingface.github.io/candle/), [\[docs.rs\]](https://docs.rs/candle-core)

***

# The Candle-specific nuance that matters a lot

## 1) Non-quantized models

For non-quantized Candle models like Gemma-style models, the public API pattern strongly suggests **internal mutable cache + explicit reset hook**: `forward(&mut self, ..., seqlen_offset)` plus `clear_kv_cache(&mut self)`. [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html)

That means for your own server, one simple pattern is:

*   one model instance **per active conversation/session**, or
*   clone the model per session if you need multiple concurrent conversations.

A Candle maintainer explicitly noted in an issue that Gemma and “most transformer based models in candle” have mutable state because of the KV cache, and that to serve different users you should clone the model so each clone has a separate cache; they also said cloning is cheap because the weights are shared. [\[github.com\]](https://github.com/huggingface/candle/issues/1918)

## 2) Quantized models

For upstream quantized Candle models, the situation is trickier because the cache is embedded in the model implementation itself, and the `candle_pipelines_models` crate exists specifically to fix that by moving cache state out of the model and into an explicit cache object. [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

So if you are using **quantized Gemma 3 via upstream Candle**, the answer is likely:

*   **yes, you are using an internal cache**,
*   **no, you probably do not have good direct observability of its size from the public API**, and
*   **your control options are more limited unless you patch/fork the model implementation or switch to an external-cache design**. [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

***

# What I would recommend you do

## Recommendation A — Treat cache length as first-class state in *your* session object

Even if Candle does not expose “KV cache bytes used”, you usually know the **cached sequence length** because you are the one feeding prompts/tokens and incrementing offsets. That means you can keep your own session struct like:

```rust
struct SessionState {
    id: Uuid,
    model: SessionModelHandle,   // maybe cloned model, or Arc + mutex
    cached_tokens: usize,
    last_access: Instant,
    history_tokens: Vec<u32>,
    // optional:
    estimated_kv_bytes: usize,
}
```

This is practical because cache size scales with cached sequence length, layers, KV heads, and head dimension, so an estimate based on model config is already very useful operationally. [\[deepwiki.com\]](https://deepwiki.com/huggingface/transformers/4.1-generation-configuration-and-strategies), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

If you control the `seqlen_offset` you pass into `forward`, then you already have a very good proxy for “how full is this session’s cache?”. [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs), [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html)

***

## Recommendation B — Impose your own hard cap

Do **not** let sessions grow forever. The Candle Gemma 3 quantized example hard-caps prompt length with `max_seq_len = 8192` and truncates old prompt tokens before generation. Even if you choose a different number, that is the right architecture: enforce a max cached/prompt token budget yourself. [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

A good first-pass policy for a local single-user/small-multi-session system is:

*   `soft_limit_tokens`: when exceeded, trigger compaction / summarization,
*   `hard_limit_tokens`: refuse further growth until compaction/reset happens,
*   `idle_ttl`: if no prompt for N minutes, clear cache or drop the session. [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs), [\[huggingface.co\]](https://huggingface.co/docs/transformers/generation_strategies)

***

## Recommendation C — Your summarize / clear / warm-up idea is good

What you described is a **completely sane application-layer strategy**:

1.  Keep full chat history externally in your app state.

2.  When the session grows too long or too idle, summarize older turns into a compact state.

3.  Clear the model cache (or recreate the session model instance).

4.  Rebuild the prompt as something like:

    *   system prompt
    *   memory summary
    *   last 1–3 raw turns
    *   current user prompt

5.  Prefill once on that compact prompt to “warm” the cache again.

6.  Continue normal decoding.

This is essentially the same tradeoff people use in long-running assistant systems: summarization/compaction spends extra tokens/compute, but prevents unbounded cache and context growth. The broader ecosystem also uses techniques like sliding windows, summarization, or cache offloading/compaction to manage long contexts. [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md), [\[cogsciprag.github.io\]](https://cogsciprag.github.io/Understanding-LLMs-course/tutorials/03a-tokenization-transformers.html), [\[huggingface.co\]](https://huggingface.co/docs/transformers/generation_strategies)

***

# How I would implement this in Rust/Candle

## If you are using non-quantized Gemma-style Candle model

This is the easiest path, because `clear_kv_cache()` is part of the public API for Candle’s non-quantized Gemma model. [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html)

Your control loop can simply do:

```rust
if session.cached_tokens > SOFT_LIMIT || session.last_access.elapsed() > IDLE_TTL {
    let summary = summarize_history(&session.history_text)?;
    session.model.clear_kv_cache(); // explicit reset
    session.history_tokens = tokenize(compose_compact_prompt(summary, recent_turns))?;
    prefill(&mut session.model, &session.history_tokens)?;
    session.cached_tokens = session.history_tokens.len();
    session.last_access = Instant::now();
}
```

That is exactly the shape of control flow I would choose. The only thing to watch is that your “prefill” logic should pass the whole compact prompt (or split prompt tokens with increasing offsets) consistently with the model’s API. [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

***

## If you are using upstream quantized Gemma 3

This is the awkward case. Because the cache is embedded in the quantized model object, your options are roughly:

### Option 1 — Recreate / clone a fresh session model

If you can cheaply clone or rebuild the quantized model object for each session reset, then “clearing cache” can just mean **drop old model instance, create a fresh one, prefill compact prompt**. The Gemma issue discussion suggests cloning in Candle is cheap in the non-quantized case because weights are shared; for quantized upstream, the external-cache critique from `candle_pipelines_models` suggests you may want to validate how expensive cloning really is in your exact setup. [\[github.com\]](https://github.com/huggingface/candle/issues/1918), [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

### Option 2 — Patch/fork to externalize cache

If you want serious control/observability, this is probably the best engineering direction. The `candle_pipelines_models` crate already documents the exact design change: instead of `forward(&mut self, input, offset)`, use a model whose weights are shareable and whose `Cache` is a separate object per conversation. That gives you explicit cache lifecycle management and makes your monitoring process much cleaner. [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

### Option 3 — Keep upstream, track only logical size

If you do not want to fork, then just track:

*   `cached_tokens`
*   `history token count`
*   `last access`
*   estimated KV bytes from config

and trigger **model recreation + compact prefill** when thresholds are crossed. This is less elegant, but absolutely workable. [\[deepwiki.com\]](https://deepwiki.com/huggingface/transformers/4.1-generation-configuration-and-strategies), [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs)

***

# A practical control policy that would work well on your M4 Max 36GB

Here is a pragmatic design I’d use for a local Mac setup:

## Session metadata you maintain yourself

```rust
struct SessionMeta {
    cached_tokens: usize,
    last_access: Instant,
    total_generated_tokens: usize,
    estimated_kv_bytes: usize,
    summary_version: u64,
}
```

## Thresholds

*   `IDLE_CLEAR_AFTER = 10–30 min`
*   `SOFT_LIMIT_TOKENS = 4k–8k` (or lower if you want snappy behavior)
*   `HARD_LIMIT_TOKENS = 8k–16k` depending on model + quantization + memory headroom
*   `MAX_ACTIVE_SESSIONS = small` if you are running fully local and care about latency. [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

## Actions

*   **idle session** → clear or drop cache/model instance
*   **soft limit exceeded** → summarize + clear + compact prefill
*   **hard limit exceeded** → force compaction before next user turn
*   **global memory pressure** → evict least-recently-used session caches first

This is basically “TTL + LRU + compaction”, which is a very reasonable local serving strategy. Continuous batching/paged KV is what bigger inference servers do, but for your single-machine Rust server you can get very far with simple lifecycle rules. [\[huggingface.co\]](https://huggingface.co/docs/transformers/generation_strategies), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

***

# My direct answer to your questions

## “I’m not sure what KV cache I’m using there”

You are **very likely using an internal per-model KV cache** in Candle, especially if your API involves `&mut self` on `forward(...)` and an offset/position parameter. For non-quantized Gemma, Candle’s docs explicitly expose `clear_kv_cache()`, and Candle maintainers describe these models as having mutable KV-cache state. For upstream quantized Gemma 3, there is evidence that the cache is embedded inside the model object, not externalized. [\[docs.rs\]](https://docs.rs/candle-transformers/latest/candle_transformers/models/gemma/struct.Model.html), [\[github.com\]](https://github.com/huggingface/candle/issues/1918), [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

## “Will it likely keep growing the KV cache until memory runs out?”

If you keep extending the same conversation/session and never reset or truncate, **yes, it can keep growing with sequence length**. I would not assume automatic eviction in Candle Gemma 3. The quantized Gemma example itself manually trims prompt length to a max sequence length, which strongly suggests you should manage this yourself. [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md), [\[github.com\]](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs), [\[mbrenndoerfer.com\]](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization)

## “Can I monitor and clear or summarize?”

Yes — and I think you **should**. Even if Candle does not expose direct cache-byte introspection for your specific model, you can reliably track `cached_tokens`, estimate KV memory from model config, enforce TTL/LRU policies, and do **summary → clear/recreate → compact prefill** as a control loop. That is a sound design. [\[deepwiki.com\]](https://deepwiki.com/huggingface/transformers/4.1-generation-configuration-and-strategies), [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/), [\[github.com\]](https://github.com/huggingface/blog/blob/main/continuous_batching.md)

***

# My strongest recommendation

If this is becoming a real inference system and not just a one-off experiment, I would seriously consider **moving to an external-cache design** for Candle instead of relying on embedded KV state inside the model. The existence of `candle_pipelines_models` is basically a sign that other Rust/Candle users hit the same wall and wanted per-conversation cache control without cloning whole mutable model objects. [\[docs.rs\]](https://docs.rs/candle-pipelines-models/latest/candle_pipelines_models/)

If you want, I can do one of two very practical next steps for you:

1.  **Sketch a Rust session manager design** (`Session`, `SessionStore`, `Compactor`, `PrefillWorker`) tailored to Candle, or
2.  **Help you inspect your exact Candle code path** and infer whether you’re on non-quantized Gemma 3, quantized GGUF Gemma 3, or a custom wrapper — and from that tell you the cleanest cache-control approach.
