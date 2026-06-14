use async_trait::async_trait;
use nexo_core::{Result, Tool, ToolCall, ToolResult, ToolResultContent, ToolResultStatus};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Arguments for the `echo` tool, which echoes back the input text.
#[derive(Deserialize, Serialize, JsonSchema)]
pub struct EchoArgs {
    /// The text to echo back
    #[schemars(description = "The text to echo back")]
    pub input: String,
}

/// Echoes an `input` string argument back to the caller.
#[derive(Debug, Default)]
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    type Args = EchoArgs;

    fn name(&self) -> &str {
        "echo.run"
    }

    fn description(&self) -> &str {
        "Echoes the input back as output"
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        Ok(ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status: ToolResultStatus::Success,
            content: ToolResultContent::Text(args.input),
        })
    }
}
