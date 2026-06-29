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
* `nexo-ai`: library-first local inference runtime; maps `nexo-core` requests and
  responses to `mistralrs-core` and loads local model configs.
* `nexo-gateway`: Gateway daemon (`nexo` binary); owns WebSocket routing, SQLite
  state, sessions, runs, rounds, cron, prompt collections, tool routing, and
  gateway-native tools.
* `nexo-node`: execution node binary; connects to the Gateway, registers local tools,
  manages loaded local models through `nexo-ai`, and handles `run.round`,
  `image.analyze`, and forwarded tool requests.
* `nexo-client`: terminal/user client binary and schema command for speaking the
  Gateway protocol.
* `nexo-tools/nexo-echo`: sample `echo.run` and `ping` tools using `nexo-core`
  contracts.
* `nexo-tools/nexo-notes`: gateway-native note tools backed by git storage.
* `nexo-tools/nexo-io`: gateway-native filesystem, shell, web-fetch, and text
  transformation tools.
* `nexo-tools/plugins/epub-extractor`: standalone EPUB extraction CLI/library.
* `nexo-tools/plugins/game-extractor`: standalone game asset extraction and analysis
  CLI/library.
* `integration-tests`: cross-crate protocol and Gateway/node integration tests.

