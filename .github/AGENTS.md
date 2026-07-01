# NEXO

NEXO is a Rust-first local agent platform: a WebSocket Gateway coordinates user
clients, execution nodes, local model inference, tools, sessions, prompt
collections, cron jobs, and git-backed notes.

## Crates

* `shared/nexo-core`: canonical Rust domain contracts for IDs, messages, model
  descriptors, inference requests/responses, run/round types, and tools.
* `shared/nexo-ws-schema`: JSON WebSocket protocol frames, methods, events, payload
  structs, wire casing, and schema generation.
* `shared/nexo-ws-client`: reusable async WebSocket connection and handshake helpers
  for clients and nodes.
* `nexo-ai`: library-first local inference runtime; Provides single entrypoint for loading and running local models, including LLMs, embeddings, and image models.
* `nexo-gateway`: Gateway daemon (`nexo` binary); owns WebSocket routing, SQLite
  state, sessions, runs, rounds, cron, prompt collections, tool routing, and
  gateway-native tools.
* `nexo-node`: execution node binary; connects to the Gateway, registers local tools and models, and executes inference and tool calls from the Gateway.
* `nexo-user`: terminal/user client binary for the nexo Gateway; provides a TUI for running inference and tools.
* `nexo-tools/nexo-echo`: simple `echo.run` tool using `nexo-core` contracts.
* `nexo-tools/nexo-notes`: gateway-native note tools backed by git storage.
* `nexo-tools/nexo-io`: gateway-native filesystem, shell, web-fetch, and text transformation tools.
* `nexo-tools/plugins/epub-extractor`: EPUB extraction CLI/library. 
* `nexo-tools/plugins/game-extractor`: game asset extraction and analysis CLI/library.

