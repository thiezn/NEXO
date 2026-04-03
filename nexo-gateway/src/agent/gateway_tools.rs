use nexo_spec::tool::Tool;
use nexo_ws_schema::ToolEntry;
use std::collections::HashMap;
use std::sync::Arc;

/// Executor for tools that run inside the gateway process (no node forwarding).
///
/// Gateway-native tools (like notes) are registered here at startup and appear
/// in the tool catalog alongside node-hosted tools. The agent loop checks this
/// executor first before forwarding tool calls to nodes.
pub struct GatewayToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl GatewayToolExecutor {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a gateway-native tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name (for execution). Returns a cloneable `Arc`.
    pub fn get_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Build `ToolEntry` descriptors for all registered gateway-native tools.
    /// These are always available (not dependent on a connected peer).
    pub fn tool_entries(&self) -> Vec<ToolEntry> {
        self.tools
            .values()
            .map(|t| {
                let spec = t.spec();
                ToolEntry {
                    name: spec.name,
                    description: spec.description,
                    source: "gateway".into(),
                    available: true,
                    parameters: Some(spec.parameters),
                }
            })
            .collect()
    }
}

impl Default for GatewayToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use nexo_spec::tool::ToolResult;

    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy.test"
        }
        fn description(&self) -> &str {
            "A test tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: true,
                output: "ok".into(),
                error: None,
            })
        }
    }

    #[test]
    fn register_and_find_tool() {
        let mut executor = GatewayToolExecutor::new();
        executor.register(Arc::new(DummyTool));
        assert!(executor.get_tool("dummy.test").is_some());
        assert!(executor.get_tool("other.tool").is_none());
    }

    #[test]
    fn tool_entries_are_always_available() {
        let mut executor = GatewayToolExecutor::new();
        executor.register(Arc::new(DummyTool));
        let entries = executor.tool_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "dummy.test");
        assert_eq!(entries[0].source, "gateway");
        assert!(entries[0].available);
    }
}
