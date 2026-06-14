//! Filesystem, shell, and web-fetch tools used by gateway-side tool execution.

/// IO tool definitions and executor implementation.
pub mod tools;
pub use tools::{BashTool, EditTool, ReadTool, WebFetchTool};

/// Shared text/content transformation helpers used by IO tools.
pub(crate) mod transform;

/// Registers all IO tools with the provided `ToolRegistry` implementation.
pub fn register_all_tools(registry: &mut nexo_core::ToolRegistry) -> nexo_core::Result<()> {
    registry.register(BashTool)?;
    registry.register(EditTool)?;
    registry.register(ReadTool)?;
    registry.register(WebFetchTool)?;
    Ok(())
}
