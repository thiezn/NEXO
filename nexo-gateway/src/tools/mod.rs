//! Tool registration and execution helpers shared across gateway components.

mod executor;
mod registry;

pub use executor::{execute_tool, tool_capability};
pub use registry::GatewayToolExecutor;
