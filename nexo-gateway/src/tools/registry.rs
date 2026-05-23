//! Gateway-local tool registry.

use nexo_spec::tool::Tool;
use nexo_ws_schema::ToolEntry;
use std::collections::HashMap;
use std::sync::Arc;

/// Stores tools that execute directly inside the gateway process.
///
/// Gateway-native tools are registered during startup and exposed in the
/// shared tool catalog alongside node-hosted tools.
pub struct GatewayToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl GatewayToolExecutor {
    /// Create an empty gateway-local tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a gateway-native tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Look up a registered tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Build tool catalog entries for every registered gateway-local tool.
    pub fn tool_entries(&self) -> Vec<ToolEntry> {
        self.tools
            .values()
            .map(|tool| ToolEntry::new(tool.spec(), "gateway", true))
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
        assert_eq!(entries[0].spec.name, "dummy.test");
        assert_eq!(entries[0].source, "gateway");
        assert!(entries[0].available);
    }
}
