use nexo_spec::tool::{Tool, ToolResult};

/// Simple echo tool for testing the node pipeline end-to-end.
pub(super) struct EchoTool;

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

/// Simple ping tool that returns `pong`.
pub(super) struct PingTool;

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
