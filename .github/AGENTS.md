## Rust Development guidelines

- Always use the latest stable rust versions
- No backwards compatibility is required, feel free to refactor and break things if it improves the codebase
- Set Rust crate dependencies only to the latest major version, and avoid setting a minor version. (e.g. use `rayon = "1"` instead of `rayon = "1.5.0"`)

## Swift Development guidelines

- Always use the latest stable Swift versions.
- Target macOS 26 and iOS26
- Use strict concurrency features and avoid using legacy concurrency patterns.
- Always use the Swift xcode MCP server to build, test and read documentation for any swift code

## CLI Tools

When building CLI tools, ensure to leverage the `cli-helpers` crate for common functionality.

Do not add any non-generic code to the `cli-helpers` crate.

