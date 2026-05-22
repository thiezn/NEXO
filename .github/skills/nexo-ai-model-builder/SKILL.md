---
name: nexo-ai-model-builder
description: Use when adding or updating a model family, runtime, or provider integration in nexo-ai. Covers manifest registration, factory wiring, Candle implementations, and OpenAI-compatible remote backends.
---

# Adding Models to nexo-ai

This skill is intentionally thin. Read the reference that matches the work you are doing instead of loading everything into context at once.

## Start Here

- `references/architecture_and_registration.md` — family layout, manifest/runtime enums, coordinator/factory wiring
- `references/openai_models.md` — OpenAI-compatible models, managed providers, speech vs chat adapters
- `references/testing.md` — integration test targets, remote smoke tests, and commands
- `references/conversation_and_templates.md` — `ChatTemplate`, `ConversationManager`, and family-local templates
- `references/tool_use.md` — tool prompt formatting and tool-call parsing
- `references/performance.md` — Candle and Metal performance pitfalls

## Current Architecture Rules

1. Work family-first, not category-first. Model code lives under `nexo-ai/src/models/<family>/`.
2. Choose the runtime up front:
  - `ModelRuntime::Candle(...)` for local inference.
  - `ModelRuntime::OpenAi { provider, model_repo }` for managed remote servers.
3. Keep family roots small. Put shared logic in `common/`; put backend-specific code in `candle/`, `openai/`, or established backend folders such as `gguf/` / `safetensors/`.
4. Registration is incomplete until every relevant layer is wired:
  - `nexo-ai/src/registry/manifest.rs`
  - `nexo-ai/src/coordinator/factory.rs`
  - `nexo-ai/src/models/mod.rs`
  - tests
5. Ignore older guidance that points at `models/multipurpose/...` or `shared/model_traits.rs`. That layout is stale.

## Workflow

1. Inspect the source model or provider.
  - Use `.claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py` for Hugging Face repos.
  - For OpenAI-compatible backends, inspect both the repo id and the request model id the server expects.
2. Pick the target family and runtime.
  - Existing family + new backend: extend the existing family module.
  - New family: add `nexo-ai/src/models/<family>/` and keep the folder small.
3. Implement the model code.
  - Candle/local: use `references/architecture_and_registration.md`.
  - OpenAI chat/tool/image: use `references/openai_models.md`.
  - OpenAI listen/talk: use `references/openai_models.md` and `nexo-ai/src/openai/speech.rs`.
4. Register the manifest and factory branch.
5. Add or update focused tests.
6. Run the smallest relevant ignored tests before moving to larger variants.

## Quick Checklist

- Manifest added with the correct `family`, `runtime`, categories, and files
- Family exported from `nexo-ai/src/models/mod.rs`
- Factory dispatch added in `nexo-ai/src/coordinator/factory.rs`
- Provider server support added if a new remote provider is involved
- Templates or tool parsing updated if the family supports Chat or Tool
- Integration tests added or updated
- Skill references updated when the architecture surface changes

## Hugging Face Inspection Commands

```bash
SCRIPT=".claude/skills/nexo-ai-model-builder/scripts/hf_downloader.py"

# 1. See directory structure first.
python3 "$SCRIPT" tree <repo_id> --pretty

# 2. Auto-detect likely components.
python3 "$SCRIPT" autodetect <repo_id> --pretty

# 3. Inspect exact files, sizes, and hashes.
python3 "$SCRIPT" inspect <repo_id> --filter "*.safetensors" --sha256 --pretty

# 4. Read config/tokenizer/preprocessing files.
python3 "$SCRIPT" config <repo_id> --all --pretty

# 5. Generate manifest code when the runtime is local.
python3 "$SCRIPT" manifest <repo_id> \
  --files "transformer/*.safetensors" "tokenizer/tokenizer.json" \
  --component-enum AiComponent --sha256 --pretty
```

Token resolution order: `HF_TOKEN` -> `hugging_token.txt` at the repo root -> `~/.nexo/hf_token.txt` -> Hugging Face CLI cached token.

For provider-managed remote models, inspect the repo for naming and metadata, but do not force the local manifest into a downloaded-file shape if the server owns downloads.
