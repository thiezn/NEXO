
Review the NEXA architecture and gateway protocol and implement the nexo-ws-schema, nexo-gateway and nexo-client.

- Details on the architecture are described @docs/src/nexo/architecture.md
- Details onto the gateway protocol are described @docs/src/nexo/gateway_protocol.md

# Crates and Files

## utl-helpers

utl-helpers is a crate to build CLI utilities. Expand the crate with helpers to generate/update a <config>.toml file through the command line. It should be completely independent from any business logic. Business logic should be implemented in the CLI tools that use this crate.

You can probably use the console crate for this functionality, but feel free to explore other options if you think they would be a better fit.

## nexo-ws-schema

Code will reside in the @shared/nexo-ws-schema/Cargo.toml crate

- The Rust struct definitions for the WebSocket protocol messages, including the `connect` handshake and any other relevant messages.
- serde json (de)serialization for these structs.
- Method to generate the full json schema, or specific sections of the schema (e.g, client, node, gateway, events) for documentation and validation purposes.
- error.rs - Expose a shared error type used by the gateway and client for WS protocol errors, including validation errors, connection errors, and message handling errors. This allows for consistent error handling across both components.

## nexo-gateway

Code will reside in the @nexo-gateway/Cargo.toml crate

- The WebSocket server implementation
- Integration with the nexo-ws-schema for message handling and validation
- Use the /cli-tool-builder to build the CLI interface for the gateway. it's config file should be set to ~/.nexo/gateway.toml
- Add a `init` command to the CLI to prepare the gateway.toml config file. Use the new helpers in utl-helpers to generate/update the config file through the command line.
- Add a `start` command to the CLI to start the gateway server, with options for configuration (e.g. port, logging level). use the gateway.toml to set defaults
- Add a `schema` generation command to the CLI, using the nexo-ws-schema helpers

## nexo-client

Code will reside in the @nexo-client/Cargo.toml crate

- The WebSocket client implementation
- Integration with the nexo-ws-schema for message handling and validation
- Use the /cli-tool-builder to build the CLI interface for the client. it's config file should be set to ~/.nexo/client.toml
- Add a `schema` generation command to the CLI, using the nexo-ws-schema helpers

# Scope

- Have a working WebSocket server and client that can perform the handshake and exchange messages according to the protocol defined in the documentation.
- Ensure that the message schemas are correctly defined and validated using the nexo-ws-schema crate.
- The CLI interfaces for both the gateway and client should allow for basic operations such as starting the server/client and sending test messages.
- Error handling and retry mechanisms for the WebSocket connections should be implemented to ensure robustness and reliability.
- Have reasonable unit test coverage for the message schemas, WebSocket server and client implementations, and error handling logic. Use `cargo llvm-cov --package nexo-client --no-cfg-coverage --skip-functions` for coverage reporting.
