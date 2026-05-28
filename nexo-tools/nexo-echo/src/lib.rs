//! Echo and ping tools implemented with `nexo-core` tool contracts.

#![forbid(unsafe_code)]

use nexo_core::{
    Error, MetadataMap, Result, ToolCall, ToolDefinition, ToolExecutionConstraints, ToolExecutor,
    ToolParallelism, ToolRegistry, ToolResult, ToolResultContent, ToolResultStatus,
    ToolSideEffectLevel,
};

/// Echoes an `input` string argument back to the caller.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoTool;

impl EchoTool {
    /// Returns the model-facing definition for the echo tool.
    #[must_use]
    pub fn definition() -> ToolDefinition {
        ToolDefinition {
            name: "echo.run".to_string(),
            description: "Echoes the input back as output".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The text to echo back"
                    }
                },
                "required": ["input"]
            }),
            contract_version: None,
            execution: read_only_execution(),
            metadata: MetadataMap::new(),
        }
    }
}

impl ToolExecutor for EchoTool {
    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        if call.name != EchoTool::definition().name {
            return Err(Error::InvalidRequest {
                message: format!("echo tool cannot execute '{}'", call.name),
            });
        }

        let input = call
            .arguments
            .get("input")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();

        Ok(ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status: ToolResultStatus::Success,
            content: ToolResultContent::Text(input),
        })
    }
}

/// Returns `pong` for connectivity checks.
#[derive(Debug, Clone, Copy, Default)]
pub struct PingTool;

impl PingTool {
    /// Returns the model-facing definition for the ping tool.
    #[must_use]
    pub fn definition() -> ToolDefinition {
        ToolDefinition {
            name: "ping".to_string(),
            description: "Returns pong - useful for testing connectivity".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            contract_version: None,
            execution: read_only_execution(),
            metadata: MetadataMap::new(),
        }
    }
}

impl ToolExecutor for PingTool {
    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        if call.name != PingTool::definition().name {
            return Err(Error::InvalidRequest {
                message: format!("ping tool cannot execute '{}'", call.name),
            });
        }

        Ok(ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status: ToolResultStatus::Success,
            content: ToolResultContent::Text("pong".to_string()),
        })
    }
}

/// Registers the echo crate tools into an existing core registry.
///
/// # Arguments
///
/// * `registry` - The registry to extend with echo and ping tools.
///
/// # Errors
///
/// Returns an error if a tool name is already registered.
pub fn register_tools(registry: &mut ToolRegistry) -> Result {
    registry.register(EchoTool::definition(), EchoTool)?;
    registry.register(PingTool::definition(), PingTool)?;
    Ok(())
}

/// Builds a registry containing the echo crate tools.
///
/// # Errors
///
/// Returns an error if built-in registration fails.
pub fn tool_registry() -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    register_tools(&mut registry)?;
    Ok(registry)
}

fn read_only_execution() -> ToolExecutionConstraints {
    ToolExecutionConstraints {
        side_effect_level: ToolSideEffectLevel::ReadOnly,
        parallelism: ToolParallelism::ParallelGlobal,
        timeout_ms: None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used)]

    use super::*;
    use nexo_core::{ToolCall, ToolCallId};
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    #[test]
    fn registry_contains_echo_and_ping() {
        let registry = tool_registry().unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.contains("echo.run"));
        assert!(registry.contains("ping"));
    }

    #[test]
    fn echo_executes_with_core_tool_call() {
        let registry = tool_registry().unwrap();
        let result = block_ready(registry.execute(ToolCall {
            id: ToolCallId::from("call-1"),
            index: 0,
            name: "echo.run".to_string(),
            arguments: serde_json::json!({"input": "hello"}),
        }))
        .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(result.content, ToolResultContent::Text("hello".to_string()));
    }

    #[test]
    fn ping_executes_with_core_tool_call() {
        let registry = tool_registry().unwrap();
        let result = block_ready(registry.execute(ToolCall {
            id: ToolCallId::from("call-1"),
            index: 0,
            name: "ping".to_string(),
            arguments: serde_json::json!({}),
        }))
        .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(result.content, ToolResultContent::Text("pong".to_string()));
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
