use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use crate::contracts::ToolExecutor;
use crate::error::{Error, Result};
use crate::tools::{ToolCall, ToolDefinition, ToolResult};

type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolResult>> + 'a>>;

trait RegisteredTool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    fn execute(&self, call: ToolCall) -> ToolFuture<'_>;
}

struct RegisteredExecutor<T> {
    definition: ToolDefinition,
    executor: T,
}

impl<T> RegisteredTool for RegisteredExecutor<T>
where
    T: ToolExecutor + 'static,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute(&self, call: ToolCall) -> ToolFuture<'_> {
        Box::pin(self.executor.execute(call))
    }
}

/// Generic in-memory registry for tools defined with `nexo-core` contracts.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Box<dyn RegisteredTool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a concrete executor with its model-facing tool definition.
    ///
    /// # Arguments
    ///
    /// * `definition` - The tool schema and execution metadata to advertise.
    /// * `executor` - The concrete executor that handles calls for this definition.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name is empty or already registered.
    pub fn register<T>(&mut self, definition: ToolDefinition, executor: T) -> Result
    where
        T: ToolExecutor + 'static,
    {
        let name = definition.name.trim();
        if name.is_empty() {
            return Err(Error::InvalidRequest {
                message: "tool name must not be empty".to_string(),
            });
        }

        let name = name.to_string();
        if self.tools.contains_key(&name) {
            return Err(Error::InvalidState {
                message: format!("tool '{name}' is already registered"),
            });
        }

        self.tools.insert(
            name,
            Box::new(RegisteredExecutor {
                definition,
                executor,
            }),
        );
        Ok(())
    }

    /// Returns the registered definition for a tool name.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name to look up.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<ToolDefinition> {
        self.tools.get(name).map(|tool| tool.definition().clone())
    }

    /// Returns all registered tool definitions in deterministic name order.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect()
    }

    /// Returns all registered tool names in deterministic order.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Returns capability labels derived from the prefix before `.` in each tool name.
    #[must_use]
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
    #[must_use]
    pub fn capabilities_and_commands(&self) -> (Vec<String>, Vec<String>) {
        (self.capability_names(), self.tool_names())
    }

    /// Returns whether a tool name is registered.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name to check.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns whether the registry has no tools.
    #[must_use]
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

impl ToolExecutor for ToolRegistry {
    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        let tool_name = call.name.clone();
        self.try_execute(call)
            .await?
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("tool '{tool_name}' is not registered"),
            })
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tool_names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used)]

    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    use crate::tools::{
        ToolExecutionConstraints, ToolParallelism, ToolResultContent, ToolResultStatus,
        ToolSideEffectLevel,
    };
    use crate::{MetadataMap, ToolCallId};

    use super::*;

    #[derive(Clone)]
    struct StaticTool;

    impl ToolExecutor for StaticTool {
        async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult {
                tool_call_id: call.id,
                tool_name: call.name,
                status: ToolResultStatus::Success,
                content: ToolResultContent::Text("ok".to_string()),
            })
        }
    }

    #[test]
    fn registers_and_executes_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register(definition("dummy.test"), StaticTool)
            .unwrap();

        let result = block_ready(registry.execute(ToolCall {
            id: ToolCallId::from("call-1"),
            index: 0,
            name: "dummy.test".to_string(),
            arguments: serde_json::json!({}),
        }))
        .unwrap();

        assert_eq!(result.tool_name, "dummy.test");
        assert_eq!(registry.tool_names(), vec!["dummy.test".to_string()]);
        assert_eq!(registry.capability_names(), vec!["dummy".to_string()]);
    }

    #[test]
    fn try_execute_returns_none_for_unknown_tool() {
        let registry = ToolRegistry::new();
        let result = block_ready(registry.try_execute(ToolCall {
            id: ToolCallId::from("call-1"),
            index: 0,
            name: "missing.test".to_string(),
            arguments: serde_json::json!({}),
        }))
        .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        let mut registry = ToolRegistry::new();
        registry
            .register(definition("dummy.test"), StaticTool)
            .unwrap();

        let error = registry
            .register(definition("dummy.test"), StaticTool)
            .unwrap_err();

        assert!(matches!(error, Error::InvalidState { .. }));
    }

    fn definition(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "Test tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
                timeout_ms: None,
            },
            metadata: MetadataMap::new(),
        }
    }

    fn block_ready<F>(future: F) -> F::Output
    where
        F: Future,
    {
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);

        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("test future unexpectedly pending"),
        }
    }

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }
}
