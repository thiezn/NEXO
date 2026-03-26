---
name: nexo-ai-model-builder
description: Use when adding a new AI model to the nexo-ai framework. Covers model trait implementation, registry registration, inference pipeline setup with Candle, and HuggingFace model inspection.
---

# Adding a New Model to nexo-ai

## Model Categories

| Category | Trait | Input -> Output |
|----------|-------|-----------------|
| Chat | `ChatModel` | text -> text |
| Tool | `ToolModel` | text + tool specs -> structured tool calls |
| Image | `ImageModel` | image + text -> text |
| Listen | `ListenModel` | audio -> text |
| Talk | `TalkModel` | text -> audio |
| Imagine | `ImagineModel` | text -> image |

Models serving multiple categories go under `models/multipurpose/`.

## Workflow

### 1. Inspect the HuggingFace repo

Use the bundled inspector script to understand the repo structure before writing any code:

```bash
SCRIPT=".claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py"

# Step 1: See directory structure (always start here)
python3 $SCRIPT tree <repo_id> --pretty

# Step 2: Auto-detect components (transformer, vae, tokenizer, etc.)
python3 $SCRIPT autodetect <repo_id> --pretty

# Step 3: Get exact file sizes and SHA-256 hashes for manifest
python3 $SCRIPT inspect <repo_id> --filter "*.safetensors" --sha256 --pretty

# Step 4: Fetch model configs (architecture, tokenizer, preprocessing)
python3 $SCRIPT config <repo_id> --all --pretty

# Step 5: Generate Rust manifest code (supports globs)
python3 $SCRIPT manifest <repo_id> \
  --files "transformer/*.safetensors" "vae/*.safetensors" "tokenizer/tokenizer.json" \
  --component-enum AiComponent --sha256 --pretty
```

**Token resolution order:** `HF_TOKEN` env var -> `hugging_token.txt` at project root -> `~/.nexo/hf_token.txt` -> huggingface-cli cached token.

Always run `tree` first. Model repos vary widely -- some use subdirectories, others put everything at the root. The `autodetect` command classifies files into components and suggests `AiComponent` mappings.

### 2. Create the model directory

```
nexo-ai/src/models/<category>/<family>/
  mod.rs         # Model struct, ModelInfo impl, category trait impl
  pipeline.rs    # Inference pipeline (load, generate, sequential load-use-drop)
  template.rs    # ChatTemplate impl (if model supports chat/tool categories)
```

Follow existing models as the reference implementation. Read 2-3 existing models during planning.

### 3. Implement traits and register

Follow this sequence -- read the existing code at each location:

1. **Implement `ModelInfo` + category trait(s)** -- See `shared/model_traits.rs` and any existing model's `mod.rs`
2. **Add manifest** -- In `registry/manifest.rs`, add to `build_all_manifests()` and update the test assertion count
3. **Wire coordinator factory** -- In `coordinator/load.rs`, add family match in `create_model_slot()`
4. **Register module** -- In `models/<category>/mod.rs`, add `pub mod <family>`

### 4. Implement ChatTemplate (for chat/tool models)

Models that support Chat or Tool categories **must** implement the `ChatTemplate` trait. This is how the framework formats conversation history into model-specific prompt strings and parses tool calls from output.

See [references/conversation_and_templates.md](references/conversation_and_templates.md) for the full `ChatTemplate` trait API, `ReasoningMode` mapping, and how `ConversationManager` handles multi-turn conversation in the REPL.

See [references/tool_use.md](references/tool_use.md) for implementing `format_with_tools` and `parse_tool_calls`.

**Key files:**
- Trait definition: `shared/templates/mod.rs`
- Existing implementations: `models/multipurpose/qwen3/template.rs`, `models/multipurpose/gemma3/template.rs`

### 5. Build the inference pipeline

Check `candle-transformers` first -- it includes many architectures (parler_tts, whisper, t5, flux, llama, qwen3, siglip, etc.). Use them directly before porting from Python.

**Non-obvious patterns:**
- **Sequential load-use-drop** for multi-component pipelines: load encoder -> use -> `drop()` -> load main model. Minimizes peak memory on unified memory.
- **CPU-seeded RNG for diffusion**: Never use `Tensor::randn()` on Metal -- it's non-deterministic. Generate noise on CPU with a seeded RNG, then `.to_device()`.
- **Avoid `Clone` on weight structs**: An accidental `.clone()` silently duplicates gigabytes.

### 6. Test

Add integration tests in `tests/model_inference.rs`. Use the existing macros (`listen_test!`, `talk_test!`, `imagine_test!`, `chat_test!`, `tool_test!`, `perf_test!`). Every model must have a performance test.

See [references/testing.md](references/testing.md) for the full testing guide including attribute ordering, download workflow, and test macros.

### 7. Performance validation

See [references/performance.md](references/performance.md) for common Metal pitfalls (BF16, `.contiguous()`, debug builds) and the per-model checklist.

## Implemented Models

| Model | Family | Category | Size |
|-------|--------|----------|------|
| `parler-mini` | parler | Talk | 3.5 GB |
| `parler-large` | parler | Talk | 8.7 GB |
| `whisper-large-v3` | whisper | Listen | 2.9 GB |
| `whisper-large-v3-turbo` | whisper | Listen | 1.5 GB |
| `distil-large-v3` | whisper | Listen | 1.4 GB |
| `flux-2-klein-4b` | flux | Imagine | 22 GB |
| `flux-2-klein-9b` | flux | Imagine | 49 GB |
| `flux-2-dev` | flux | Imagine | 165 GB |
| `z-image-turbo` | z_image | Imagine | 31 GB |
| `gemma-3-4b-it` | gemma3 | Chat, Tool, Image | ~8 GB |
| `gemma-3-12b-it` | gemma3 | Chat, Tool, Image | ~24 GB |
| `gemma-3-27b-it` | gemma3 | Chat, Tool, Image | ~54 GB |
| `qwen3-4b-q5km` | qwen3 | Chat, Tool | ~2.9 GB |
| `qwen3-30b-a3b-q4km` | qwen3 | Chat, Tool | ~18.6 GB |
| `qwen3-vl-4b` | qwen3 | Chat, Tool, Image | ~3.0 GB |

## After Implementation

- Run /simplify to ensure the codebase stays clean and maintainable.
- Review this SKILL.md and make suggestions on improvements or clarifications for future model builders.
