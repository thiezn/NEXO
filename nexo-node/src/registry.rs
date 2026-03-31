use nexo_tool_spec::tool::{Tool, ToolResult};
use nexo_ws_schema::ToolSpecEntry;

/// Local tool registry holding all tools this node can execute.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    /// Create a registry with the built-in tools.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(PingTool));
        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        tracing::info!("Loaded tool: {}", tool.name());
        self.tools.push(tool);
    }

    /// Get tool specs as schema-compatible entries for `tools.register`.
    pub fn specs(&self) -> Vec<ToolSpecEntry> {
        self.tools
            .iter()
            .map(|t| {
                let spec = t.spec();
                ToolSpecEntry {
                    name: spec.name,
                    description: spec.description,
                    parameters: spec.parameters,
                }
            })
            .collect()
    }

    /// Get capability names (unique tool name prefixes) and command names for `ConnectParams`.
    pub fn capabilities_and_commands(&self) -> (Vec<String>, Vec<String>) {
        let commands: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
        let mut capabilities: Vec<String> = commands
            .iter()
            .map(|c| c.split('.').next().unwrap_or(c).to_string())
            .collect();
        capabilities.sort();
        capabilities.dedup();
        (capabilities, commands)
    }

    /// Find and execute a tool by name.
    pub async fn execute(&self, tool_name: &str, args: serde_json::Value) -> Option<ToolResult> {
        let tool = self.tools.iter().find(|t| t.name() == tool_name)?;
        match tool.execute(args).await {
            Ok(result) => Some(result),
            Err(e) => Some(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }

    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// -- Built-in tools --

/// Simple echo tool for testing the node pipeline end-to-end.
struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo.run"
    }

    fn description(&self) -> &str {
        "Echoes the input back as output"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text to echo back"
                }
            },
            "required": ["input"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let input = args
            .get("input")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        tracing::debug!("Echo tool executing with input: {input}");
        Ok(ToolResult {
            success: true,
            output: input,
            error: None,
        })
    }
}

/// Simple ping tool that returns "pong".
struct PingTool;

#[async_trait::async_trait]
impl Tool for PingTool {
    fn name(&self) -> &str {
        "ping"
    }

    fn description(&self) -> &str {
        "Returns pong - useful for testing connectivity"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            success: true,
            output: "pong".to_string(),
            error: None,
        })
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
