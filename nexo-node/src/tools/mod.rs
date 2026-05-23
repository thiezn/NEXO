//! Tool registration, built-ins, and gateway tool execution for `nexo-node`.

mod builtins;
mod gateway;
mod registry;

pub(crate) use gateway::{handle_tool_execute, register_tools};
pub use registry::ToolRegistry;
