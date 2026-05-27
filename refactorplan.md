


I’ve got enough to shape this. The old crate spans registry, download, multiple backends, image/audio, CLI, and provider-server plumbing, while the new nexo-ai is an empty stub, so a staged plan is the realistic path.

Given the current state of Cargo.toml, lib.rs, Cargo.toml, lib.rs, and the old broad surface in Cargo.toml, I would not try to reimplement the full crate in one session. A single session is realistic only for an MVP: library scaffold plus `mistralrs-core` mapping plus round/run execution for text generation and tool calling. I’m assuming you mean `mistralrs-core`, since that is the actual crate name used earlier; if you literally mean another crate, replace the dependency name in the prompts below.

One repo-specific detail to include in the first session: Cargo.toml already moved the member to shared/nexo-core, but it does not yet expose `nexo-core` in `[workspace.dependencies]`, so the agent should either add that workspace dependency or add a direct path dependency in nexo-ai. Also, the old trait matrix in model_traits.rs and types.rs should be treated as reference only, not as the new design target.

**Staged Prompts**

1. Foundation and crate scaffold (DONE)

```text
/rust should be used for this task.

Rebuild the new `nexo-ai` crate as a library-first crate around `nexo-core` and `mistralrs-core`.

Context:
- `nexo-core` lives at `shared/nexo-core`.
- The current `nexo-ai` crate is basically an empty stub.
- `nexo-ai-old` is reference material only.
- `nexo-ai` must not depend on `nexo-ws-client`, or `nexo-ws-schema`.
- `nexo-ai` must not touch websocket transport or serialization concerns.
- `nexo-node` is out of scope.
- `nexo-ai` should use the workspace `cli-helpers` crate.
- Public API should expose `nexo-core` types, not `mistralrs-core` types.

Please:
- Read the current `nexo-core` public API first.
- Read the current `nexo-ai` stub and enough of `nexo-ai-old` to avoid repeating the old abstraction mess.
- Make `nexo-ai` a library-first crate with a thin bin only if needed.
- Add the right dependencies for `nexo-core`, `mistralrs-core`, `cli-helpers`, and any minimal runtime crates actually needed.
- Define a crate-local `Error` enum and `Result<T>` alias.
- Create a clean module layout with minimal top-level modules, for example: `error`, `config`, `runtime`, `mapping`, `round`, `run`, `registry`, `cli`.
- Keep `mod.rs` files minimal and place logic in separate files.
- Add rustdoc for every public type, trait, enum, function, and parameter.
- Reexport a clean public surface from `lib.rs`.
- Do not implement websocket code, node code, or legacy provider-server plumbing.
- Do not revive the old per-category trait matrix unless there is a very strong reason.

End with:
- `cargo fmt --package nexo-ai`
- `cargo check -p nexo-ai`
- `cargo clippy -p nexo-ai -- -D warnings`
- `cargo test -p nexo-ai`

Also include a short final summary of the module tree and the intended public API.
```

2. `mistralrs-core` mapping and single-round inference

```text
/rust should be used for this task.

Implement the core adapter layer inside `nexo-ai` that maps between `nexo-core` types and `mistralrs-core` types for a single inference round.

Constraints:
- Keep all `mistralrs-core` types internal to `nexo-ai`.
- Public methods should accept and return `nexo-core` types only.
- Focus on the generation path first: text chat, streaming chunks, reasoning, finish reasons, usage, and tool calling.
- Do not add websocket code.
- Do not touch `nexo-node`.
- Use `ModelDescriptor.role_strategy` from `nexo-core` when converting messages.
- `RoleStrategy::Default` should preserve roles.
- `RoleStrategy::MergeDeveloperIntoSystem` should merge developer messages into system content before request mapping.

Please implement:
- A dedicated internal mapping layer from `nexo_core::InferenceRequest` / `GenerateRequest` / conversation types / tool types into the corresponding `mistralrs-core` request structures.
- The reverse mapping from `mistralrs-core` responses and streamed deltas back into `nexo_core::InferenceResponse` values.
- Clean handling for tool definitions, tool choice, tool call deltas, finish reasons, token usage, and inference failures.
- A public round-level API in `nexo-ai` for executing one inference round.
- Prefer implementing `nexo_core::InferenceEngine` if that fits cleanly.
- Add focused unit tests for request mapping, response mapping, role strategy handling, and tool call parsing.

Do not try to implement image, speech, download, registry parity, or old provider-server features in this prompt unless they are strictly required for the round API to compile.

End with:
- `cargo fmt --package nexo-ai`
- `cargo check -p nexo-ai`
- `cargo clippy -p nexo-ai -- -D warnings`
- `cargo test -p nexo-ai`

Summarize the public round API you introduced.
```

3. Multi-round run orchestration and tool loop

```text
Use the rust skill for this task.

Build the run orchestration layer in `nexo-ai` on top of the existing round execution layer.

Goal:
- `nexo-ai` should provide public methods to run one round and a full multi-round run.
- The public API should use `nexo-core` run and round types.
- `nexo-ai` should remain transport-agnostic and websocket-free.

Constraints:
- Use only `nexo-core` as the internal shared Nexo dependency.
- Do not import `nexo-spec`, `nexo-ws-client`, or `nexo-ws-schema`.
- Do not touch `nexo-node`.
- `ToolExecutor` in `nexo-core` is static-dispatch oriented; use generic type parameters instead of trait objects where appropriate.

Please implement:
- A run orchestrator that can execute one or more rounds until completion, failure, cancellation, or a max-round limit.
- Round creation and correlation using `RunId` and `RoundId`.
- Emission of `RunEvent`, nested `RoundEvent`, `RunStatusUpdate`, and `RoundStatusUpdate`.
- Handling of model-produced tool calls by invoking the generic `ToolExecutor`, appending tool results to the conversation, and continuing into the next round.
- Clean handling of `ToolParallelism` metadata from `nexo-core`. It is acceptable to treat it as scheduling metadata only for now if fully honoring it would widen scope too much, but be explicit in code and docs.
- Small orchestration config types if needed, but keep shared protocol concepts in `nexo-core` and crate-local operational knobs in `nexo-ai`.
- Focused tests for multi-round tool loops, event ordering, role-strategy-aware follow-up rounds, and failure propagation.

Do not add websocket framing, node-specific serialization, or old remote-provider management layers.

End with:
- `cargo fmt --package nexo-ai`
- `cargo check -p nexo-ai`
- `cargo clippy -p nexo-ai -- -D warnings`
- `cargo test -p nexo-ai`

Summarize the run API and the main orchestration types.
```

4. Thin CLI and final cleanup

```text
Use the rust skill for this task.

Add a thin developer-facing CLI surface to `nexo-ai` using the workspace `cli-helpers` crate, while keeping the crate library-first.

Goal:
- The library should stay the primary product.
- The binary should exist only as a thin debugging and local-development shell around the public `nexo-ai` API.

Constraints:
- No websocket code.
- No `nexo-node` changes.
- No dependency on `nexo-spec`, `nexo-ws-client`, or `nexo-ws-schema`.
- Keep `mistralrs-core` hidden behind the `nexo-ai` public API.

Please implement:
- A minimal CLI using `cli-helpers` that can run a single inference round and a full run from local input.
- Streaming or buffered display of `nexo-core` events for debugging.
- Thin argument parsing and startup code only; business logic must stay in the library.
- Final dependency cleanup so `nexo-ai` only pulls in what it actually uses.
- Final rustdoc cleanup and public reexports review.
- A short note in the final summary about what is intentionally deferred to `nexo-node`.

End with:
- `cargo fmt --package nexo-ai`
- `cargo check -p nexo-ai`
- `cargo clippy -p nexo-ai -- -D warnings`
- `cargo test -p nexo-ai`

List any intentionally deferred areas, especially node integration, websocket transport, and unsupported modalities.
```

**Single-Session MVP**
If you want to try one session anyway, I would scope it tightly to an MVP and use this instead of trying for full crate parity from Cargo.toml or manifest.rs:

```text
Use the rust skill for this task.

Rebuild `nexo-ai` as a clean library-first crate around `nexo-core` and `mistralrs-core`, but only implement the MVP in this session.

MVP scope only:
- library-first `nexo-ai`
- dependency wiring for `nexo-core`, `mistralrs-core`, `cli-helpers`, and minimal runtime crates
- crate-local `Error` and `Result`
- clean module layout
- internal mapping between `nexo-core` and `mistralrs-core`
- single-round inference API
- multi-round run API with tool loop support
- role strategy handling from `nexo-core`
- tests for mapping and multi-round orchestration
- thin CLI only if it is cheap after the library is stable

Out of scope for this session:
- websocket transport
- `nexo-node`
- `nexo-spec`
- `nexo-ws-client`
- `nexo-ws-schema`
- download/pull flows
- old MLX/OpenAI provider-server management
- image, speech, and other non-text modalities unless they fall out almost for free
- broad feature parity with `nexo-ai-old`

Important architectural rules:
- public API must expose `nexo-core` types only
- keep `mistralrs-core` internal
- do not recreate the old trait matrix from the old crate
- keep module boundaries clean and documented
- use static dispatch where the shared traits are designed for it
- follow workspace Rust conventions and rustdoc requirements

Validation required:
- `cargo fmt --package nexo-ai`
- `cargo check -p nexo-ai`
- `cargo clippy -p nexo-ai -- -D warnings`
- `cargo test -p nexo-ai`

Finish by summarizing:
- the final module tree
- the public API for running a round and a run
- what was intentionally left for later sessions
```

If you want, I can also turn these into a tighter 2-prompt version: `MVP implementation` and `polish + CLI`, which may be the best balance between momentum and scope.
