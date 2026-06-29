use super::Tool;
use crate::error::{Error, Result};
use crate::tools::{ToolCall, ToolDefinition, ToolResult};
use async_trait::async_trait;
use std::collections::BTreeMap;

/// RegisteredTool trait that will allow us to store heterogeneous tools
/// in the registry while still being able to call their execute method.
///
/// The reason this is needed as each Tool trait implements an associated type Args,
/// which prevents us from using a simple Box<dyn Tool>.
///
/// This trait does introduce some duplication, but it allows us to keep the
/// ergonomics of defining tools with the Tool trait. Since this code here
/// is not meant to be used by end-users, we can afford to have some internal complexity
/// in order to provide a better experience for tool authors.
#[async_trait]
pub trait RegisteredTool: Send + Sync {
    /// Returns the full tool definition, which includes the name, description, parameters, contract version, and execution constraints.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the given call, returning a ToolResult or an error if execution fails.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
}

#[async_trait]
impl<T> RegisteredTool for T
where
    T: Tool + Send + Sync,
{
    /// Returns the full tool definition, which includes the name, description, parameters, contract version, and execution constraints.
    fn definition(&self) -> ToolDefinition {
        Tool::definition(self)
    }

    /// Executes the tool with the given call, returning a ToolResult or an error if execution fails.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        Tool::execute(self, call).await
    }
}

/// Generic in-memory registry for tools defined with `nexo-core` contracts.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Box<dyn RegisteredTool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool with the registry.
    pub fn register(&mut self, tool: impl Tool + 'static) -> Result {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(Error::InvalidState {
                message: format!("tool '{name}' is already registered"),
            });
        }

        self.tools.insert(name, Box::new(tool));
        Ok(())
    }

    /// Returns the registered definition for a tool name.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name to look up.
    pub fn definition(&self, name: &str) -> Option<ToolDefinition> {
        self.tools.get(name).map(|tool| tool.definition().clone())
    }

    /// Returns all registered tool definitions in deterministic name order.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Returns all registered tool names in deterministic order.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Returns capability labels derived from the prefix before `.` in each tool name.
    pub fn capability_names(&self) -> Vec<String> {
        let mut capabilities = self
            .tools
            .keys()
            .map(|name| name.split('.').next().unwrap_or(name).to_string())
            .collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        capabilities
    }

    /// Returns the advertised capability labels and tool names.
    pub fn capabilities_and_tool_names(&self) -> (Vec<String>, Vec<String>) {
        (self.capability_names(), self.tool_names())
    }

    /// Returns whether a tool name is registered.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name to check.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns whether the registry has no tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Executes a tool call when the target tool is registered.
    ///
    /// # Arguments
    ///
    /// * `call` - The concrete tool call to execute.
    ///
    /// # Errors
    ///
    /// Returns any error produced by the concrete tool executor.
    pub async fn try_execute(&self, call: ToolCall) -> Result<ToolResult> {
        let Some(tool) = self.tools.get(&call.name) else {
            return Err(Error::ToolNotFound {
                name: call.name.clone(),
            });
        };

        tool.execute(call).await
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tool_names())
            .finish()
    }
}
