//! Filesystem, shell, and web-fetch tools used by gateway-side tool execution.

/// IO tool definitions and executor implementation.
pub mod tools;
pub use tools::{BashTool, EditTool, ReadTool, WebFetchTool};

/// Shared text/content transformation helpers used by IO tools.
pub(crate) mod transform;
