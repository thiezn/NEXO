
Build modules within nexo-node that can call various local running inference runtimes for different models and modalities. The goal is for the nexo-node crate to be able to invoke various local model servers or cli's.

The Nexo node will receive instructions for load/unload models and running inference on them. We will implement this coordination in a later stage.

For now, Review the planned models and HTTP/cli services they will use and create the required reqwest tokio calls for inference. Make sure the calls can leverage tokio properly for async execution.

Regarding qwen-image-mpds, we will use it as a cli and spawn it as a subprocess from Rust, passing the required arguments and reading the output. Make sure we can call into it using tokio async as well. The cli should be hidden from the user.

Create an abstraction in @base.rs which will be the only public interface in the inference_clients module. This will have the required functions to call into the various services and cli's for inference. The internal implementation should be hidden from the user in the other inference_clients implementations.


# Overview of the models and runtimes

*   **`llama-server`** for your main GGUF reasoning model (simple, OpenAI-compatible HTTP). [\[huggingface.co\]](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
*   **`whisper-server`** from `whisper.cpp` for STT (simple local HTTP). `whisper.cpp` has a dedicated HTTP server and Apple-Silicon-first Metal acceleration. [\[deepwiki.com\]](https://deepwiki.com/ggml-org/whisper.cpp/3.2-http-server), [\[notebookcheck.net\]](https://www.notebookcheck.net/Apple-M1-Max-Processor-Benchmarks-and-Specs.579971.0.html)
*   **`mlx-tts-server`** for Qwen3-TTS on MLX (OpenAI-compatible TTS HTTP). It is specifically built for Apple Silicon + Qwen3-TTS. [\[pypi.org\]](https://pypi.org/project/mlx-tts-server/)
*   **`vllm-mlx`** for your image-analysis VLM (OpenAI-compatible HTTP, multimodal on Apple Silicon). [\[github.com\]](https://github.com/waybarrios/vllm-mlx), [\[pypi.org\]](https://pypi.org/project/vllm-mlx/)
*   **`qwen-image-mps` as a subprocess/CLI** for image generation/editing, because its Apple-Silicon path is strong but it is primarily a CLI package, not a production web server. That actually gives you the **least infrastructure** for the image side. [\[pypi.org\]](https://pypi.org/project/qwen-image-mps/), [\[github.com\]](https://github.com/ivanfioravanti/qwen-image-mps)

***

# Final model/runtime/call table

| Model                                      | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Best runtime on your M1 Max 64 GB                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | How your Rust coordinator should call it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen3.5-35B-A3B (GGUF)**                 | **Main reasoning + tool-calling + summarization** model. It fits well on a 64 GB Mac in 4-bit quantization and is one of the strongest local all-rounders for long context and agent/tool workflows. [\[huggingface.co\]](https://huggingface.co/hexgrad/Kokoro-82M/blob/3f2d7aa0c47bc8cbc19a54ae757a7b272e2abdcd/README.md), [\[sitepoint.com\]](https://www.sitepoint.com/local-llms-apple-silicon-mac-2026/)                                                                                                                                                  | **`llama-server`** via `llama.cpp`, because it gives you a lightweight local inference binary with OpenAI-compatible endpoints and strong Apple Silicon / Metal support through the `llama.cpp` ecosystem. [\[huggingface.co\]](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [\[huggingface.co\]](https://huggingface.co/hexgrad/Kokoro-82M/blob/3f2d7aa0c47bc8cbc19a54ae757a7b272e2abdcd/README.md)                                                                                                                                                         | **HTTP** from Rust using `reqwest` to `http://127.0.0.1:<port>/v1/chat/completions` (or `/v1/responses` if you standardize there). This is the cleanest and lowest-friction option. [\[huggingface.co\]](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [\[github.com\]](https://github.com/hexgrad/kokoro/releases)                                                                                                                                                                              |
| **Qwen-Image-2512 + Qwen-Image-Edit-2511** | **Text-to-image + image-to-image + LoRA-capable art stack** for your adventure-game pixel-art workflow. The Apple-Silicon ecosystem around `qwen-image-mps` explicitly supports generation, editing, Lightning LoRA, and custom LoRA models. [\[deepwiki.com\]](https://deepwiki.com/ggml-org/llama.cpp/5.2-http-server), [\[deepwiki.com\]](https://deepwiki.com/jina-ai/llama.cpp/6-http-server-and-apis), [\[github.com\]](https://github.com/ivanfioravanti/qwen-image-mps), [\[deepwiki.com\]](https://deepwiki.com/ivanfioravanti/qwen-image-mps) | **`qwen-image-mps`** (MPS / Diffusers CLI) is still the best Apple-native runtime path for these Qwen image models on a Mac. It auto-selects MPS, supports fast LoRA modes, and supports custom LoRA loading. [\[pypi.org\]](https://pypi.org/project/qwen-image-mps/), [\[github.com\]](https://github.com/ivanfioravanti/qwen-image-mps)                                                                                                                                                                                                                 | **Spawn as a subprocess** from Rust with `std::process::Command` / `tokio::process::Command`, pass prompt/input image paths/LoRA args, and read the output file path from stdout or a deterministic output directory. This is the **minimal-infrastructure** option because `qwen-image-mps` is primarily a CLI, not a dedicated server. [\[pypi.org\]](https://pypi.org/project/qwen-image-mps/), [\[github.com\]](https://github.com/ivanfioravanti/qwen-image-mps)                              |
| **Whisper large-v3-turbo**                 | **Speech-to-text** for English + Dutch, with an excellent speed/quality tradeoff. Whisper large-v3-turbo is the faster pruned variant, and Whisper supports **99 languages**, including Dutch. [\[docs.sglang.io\]](https://docs.sglang.io/basic_usage/qwen3_5.html), [\[sebastianraschka.com\]](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/)                                                                                                                                                                                                     | **`whisper.cpp`** with **`whisper-server`**. `whisper.cpp` is Apple-Silicon-first with ARM NEON, Accelerate, Metal, and Core ML support, and it includes an HTTP server for transcription. [\[discussion....apple.com\]](https://discussions.apple.com/thread/255888747), [\[notebookcheck.net\]](https://www.notebookcheck.net/Apple-M1-Max-Processor-Benchmarks-and-Specs.579971.0.html), [\[deepwiki.com\]](https://deepwiki.com/ggml-org/whisper.cpp/3.2-http-server), [\[github.com\]](https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md) | **HTTP multipart upload** from Rust to `POST /inference` on `whisper-server`; send audio as `multipart/form-data`, optionally ask for JSON / text / SRT / VTT. This keeps your integration tiny and avoids writing your own wrapper. [\[deepwiki.com\]](https://deepwiki.com/ggml-org/whisper.cpp/3.2-http-server), [\[github.com\]](https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md), [\[mozilla-ai.github.io\]](https://mozilla-ai.github.io/llamafile/whisperfile/server/) |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice**        | **Text-to-speech** with better quality/control than tiny models, using preset voices and instruction control. The Qwen3-TTS family is multilingual, streaming-capable, and optimized for low latency; the 1.7B line is the quality-focused choice. [\[clawdbook.org\]](https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026), [\[groundy.com\]](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)                                                                                       | **MLX** via **`mlx-tts-server`** (backed by `mlx-audio`). This is the cleanest Apple-Silicon server path for Qwen3-TTS: it is OpenAI-compatible and explicitly supports MLX Qwen3-TTS model IDs, including 1.7B CustomVoice variants. [\[pypi.org\]](https://pypi.org/project/mlx-tts-server/), [\[github.com\]](https://github.com/Blaizzy/mlx-audio)                                                                                                                                                                                                    | **HTTP** from Rust to `POST /v1/audio/speech` with OpenAI-style JSON (`model`, `input`, `voice`, `response_format`). If you later need cloning/design, `mlx-tts-server` also exposes those flows. [\[pypi.org\]](https://pypi.org/project/mlx-tts-server/)                                                                                                                                                                                                                                            |
| **Qwen3.5-9B**                             | **Image analysis + prompt generation** model: describe images, infer scene structure, and generate high-quality prompts for training / LoRA dataset refinement. Qwen positions Qwen3.5 as a unified vision-language foundation with strong visual understanding. [\[linkedin.com\]](https://www.linkedin.com/posts/thenextgentechinsider_qwen35-mlxquantization-gguf-activity-7440499801231745024-d-kX)                                                                                                                                                        | **`vllm-mlx`** on Apple Silicon. It is an OpenAI-compatible MLX server for text / vision / audio on Mac, built specifically to serve multimodal models locally with minimal API friction. [\[github.com\]](https://github.com/waybarrios/vllm-mlx), [\[pypi.org\]](https://pypi.org/project/vllm-mlx/)                                                                                                                                                                                                                                                     | **HTTP** from Rust to `POST /v1/chat/completions` with multimodal content (text + image). This lets your Rust side treat the VLM almost exactly like an OpenAI vision model. [\[github.com\]](https://github.com/waybarrios/vllm-mlx), [\[docs.vllm.ai\]](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)                                                                                                                                                                           |

***

# What this means in practice

## The shortest, cleanest coordinator architecture

Your Rust coordinator can standardize on just **three call styles**:

1.  **OpenAI-compatible HTTP**
    *   `llama-server` for **Qwen3.5-35B-A3B**
    *   `mlx-tts-server` for **Qwen3-TTS**
    *   `vllm-mlx` for **Qwen3.5-9B vision**
    *   all called with `reqwest` from Rust.

2.  **Simple HTTP multipart**
    *   `whisper-server` for **Whisper large-v3-turbo**. [\[deepwiki.com\]](https://deepwiki.com/ggml-org/whisper.cpp/3.2-http-server), [\[github.com\]](https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md)

3.  **Local subprocess**
    *   `qwen-image-mps` for **Qwen-Image generation/editing**. [\[pypi.org\]](https://pypi.org/project/qwen-image-mps/), [\[github.com\]](https://github.com/ivanfioravanti/qwen-image-mps)


# Recommendation on API standardization

If you want the **simplest Rust code**, normalize your internal coordinator API to:

*   **Chat / VLM:** OpenAI-style JSON
*   **TTS:** OpenAI `/v1/audio/speech`
*   **STT:** OpenAI-like transcription request or a thin internal wrapper
*   **Image generation/editing:** internal job request → subprocess → file path result

That means your Rust side can mostly be:

*   `reqwest` for text / vision / tts / stt
*   `tokio::process::Command` for image generation/editing

***

# My final operational recommendation

If I were implementing this in your codebase, I’d use these local services:

*   **Port 8001** → `llama-server` for **Qwen3.5-35B-A3B**
*   **Port 8002** → `mlx-tts-server` for **Qwen3-TTS**
*   **Port 8003** → `whisper-server` for **Whisper**
*   **Port 8004** → `vllm-mlx` for **Qwen3.5-9B image analysis**
*   **No server** for **Qwen-Image**; just launch `qwen-image-mps` on demand as a subprocess

This keeps the system **very understandable**, **very debuggable**, and avoids building wrappers you don’t actually need.

***

Think about **concrete Rust-facing interface spec** like:

*   `InferTextRequest`
*   `TranscribeAudioRequest`
*   `SynthesizeSpeechRequest`
*   `AnalyzeImageRequest`
*   `GenerateImageRequest`

plus the exact **endpoint map / CLI contract** your coordinator can implement.
