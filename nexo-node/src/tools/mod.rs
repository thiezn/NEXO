//! Tool registration, built-ins, and gateway tool execution for `nexo-node`.

mod gateway;

pub(crate) use gateway::{handle_tool_execute, register_tools};
