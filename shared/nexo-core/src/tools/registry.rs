use std::collections::HashMap;

use super::Tool;
use crate::error::{Error, Result};
use crate::tools::{ToolCall, ToolDefinition, ToolExecutionConstraints, ToolResult};

/// DynTool trait that will allow us to store heterogeneous tools
/// in the registry while still being able to call their execute method.
///
/// The reason this is needed as each Tool trail implements an associated type Args,
/// which prevents us from using a simple Box<dyn Tool>.
///
/// This macro does introduce some complexity here, but it allows us to keep the
/// ergonomics of defining tools with the Tool trait. Since this code here
/// is not meant to be used by end-users, we can afford to have some internal complexity
/// in order to provide a better experience for tool authors.
#[async_trait::async_trait]
pub trait DynTool: Send + Sync {
    /// Returns the name of the tool, which should be unique across the registry.
    fn name(&self) -> &str;

    /// Returns a human-readable description of the tool's functionality.
    fn description(&self) -> &str;

    /// Returns a JSON schema describing the tool's expected arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Returns an optional version string for the tool's contract, which can be used for compatibility checks.
    fn contract_version(&self) -> Option<&str>;

    /// Returns the execution constraints for the tool, such as side effect level and parallelism.
    fn execution_constraints(&self) -> ToolExecutionConstraints;

    /// Returns the full tool definition, which includes the name, description, parameters, contract version, and execution constraints.
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the given call, returning a ToolResult or an error if execution fails.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
}

#[async_trait::async_trait]
impl<T> DynTool for T
where
    T: Tool + Send + Sync,
{
    fn name(&self) -> &str {
        Tool::name(self)
    }
    fn description(&self) -> &str {
        Tool::description(self)
    }
    fn parameters(&self) -> serde_json::Value {
        Tool::parameters(self)
    }
    fn contract_version(&self) -> Option<&str> {
        Tool::contract_version(self)
    }
    fn execution_constraints(&self) -> ToolExecutionConstraints {
        Tool::execution_constraints(self)
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
            contract_version: self.contract_version().map(|s| s.to_string()),
            execution: self.execution_constraints(),
        }
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        Tool::execute(self, call).await
    }
}

/// Generic in-memory registry for tools defined with `nexo-core` contracts.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn DynTool>>,
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
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
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

    /// Returns the advertised capability labels and command names.
    pub fn capabilities_and_commands(&self) -> (Vec<String>, Vec<String>) {
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
    pub async fn try_execute(&self, call: ToolCall) -> Result<Option<ToolResult>> {
        let Some(tool) = self.tools.get(&call.name) else {
            return Ok(None);
        };

        tool.execute(call).await.map(Some)
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tool_names())
            .finish()
    }
}
