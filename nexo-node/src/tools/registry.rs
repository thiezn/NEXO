use super::builtins::{EchoTool, PingTool};
use nexo_spec::tool::{Tool, ToolResult};
use nexo_ws_schema::ToolSpecEntry;

/// In-memory registry of tools this node can advertise and execute.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    /// Create a registry populated with the built-in node tools.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(PingTool));
        registry
    }

    /// Register a tool so it can be advertised to and executed for the gateway.
    ///
    /// # Arguments
    ///
    /// * `tool` - The boxed tool implementation to add to the registry.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        tracing::info!("Loaded tool: {}", tool.name());
        self.tools.push(tool);
    }

    /// Return schema-compatible tool specs for the `tools.register` handshake.
    pub fn specs(&self) -> Vec<ToolSpecEntry> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }

    /// Return advertised capabilities and command names for the gateway handshake.
    pub fn capabilities_and_commands(&self) -> (Vec<String>, Vec<String>) {
        let commands: Vec<String> = self
            .tools
            .iter()
            .map(|tool| tool.name().to_string())
            .collect();
        let mut capabilities: Vec<String> = commands
            .iter()
            .map(|command| command.split('.').next().unwrap_or(command).to_string())
            .collect();

        capabilities.sort();
        capabilities.dedup();
        (capabilities, commands)
    }

    /// Execute a registered tool by name.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The name advertised to the gateway.
    /// * `args` - JSON arguments forwarded to the tool implementation.
    pub async fn execute(&self, tool_name: &str, args: serde_json::Value) -> Option<ToolResult> {
        let tool = self.tools.iter().find(|tool| tool.name() == tool_name)?;
        match tool.execute(args).await {
            Ok(result) => Some(result),
            Err(error) => Some(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error.to_string()),
            }),
        }
    }

    /// Return the number of tools currently registered on this node.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn builtin_registry_has_tools() {
        let registry = ToolRegistry::with_builtins();
        assert_eq!(registry.tool_count(), 2);
    }

    #[test]
    fn specs_returns_all_tools() {
        let registry = ToolRegistry::with_builtins();
        let specs = registry.specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "echo.run");
        assert_eq!(specs[1].name, "ping");
    }

    #[test]
    fn capabilities_and_commands() {
        let registry = ToolRegistry::with_builtins();
        let (caps, cmds) = registry.capabilities_and_commands();
        assert!(caps.contains(&"echo".to_string()));
        assert!(caps.contains(&"ping".to_string()));
        assert!(cmds.contains(&"echo.run".to_string()));
        assert!(cmds.contains(&"ping".to_string()));
    }

    #[tokio::test]
    async fn echo_tool_executes() {
        let registry = ToolRegistry::with_builtins();
        let result = registry
            .execute("echo.run", serde_json::json!({"input": "hello"}))
            .await;
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "hello");
    }

    #[tokio::test]
    async fn ping_tool_executes() {
        let registry = ToolRegistry::with_builtins();
        let result = registry.execute("ping", serde_json::json!({})).await;
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "pong");
    }

    #[tokio::test]
    async fn unknown_tool_returns_none() {
        let registry = ToolRegistry::with_builtins();
        let result = registry.execute("nonexistent", serde_json::json!({})).await;
        assert!(result.is_none());
    }
}
