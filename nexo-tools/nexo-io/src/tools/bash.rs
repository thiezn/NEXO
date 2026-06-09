use crate::transform;
use async_trait::async_trait;

use nexo_core::{
    Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Arguments for the `bash` tool, which executes a bash command with a timeout.
#[derive(Deserialize, Serialize, JsonSchema)]
pub struct BashArgs {
    /// Bash command to execute
    #[schemars(description = "Bash command to execute")]
    pub command: String,

    /// Timeout in milliseconds (default: 30000, max: 120000)
    #[schemars(description = "Timeout in milliseconds (default: 30000, max: 120000)")]
    pub timeout_ms: Option<u64>,
}

/// Executes `io.bash` tool against local filesystem
#[derive(Default)]
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    type Args = BashArgs;

    fn name(&self) -> &str {
        "io.bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command"
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args: BashArgs = self.parse_args(&call)?;

        let command = args.command;
        let timeout_ms = args.timeout_ms.unwrap_or(30_000).clamp(1_000, 120_000);
        let timeout = std::time::Duration::from_millis(timeout_ms);

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Execute the command with a timeout
        let result = tokio::time::timeout(timeout, async {
            let child = tokio::process::Command::new("bash")
                .arg("-c")
                .arg(&command)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()?;
            child.wait_with_output().await
        })
        .await;

        // Process the result
        Ok(match result {
            Ok(Ok(output)) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = transform::ansi::strip_ansi(&String::from_utf8_lossy(&output.stdout));
                let stderr = transform::ansi::strip_ansi(&String::from_utf8_lossy(&output.stderr));
                let stdout = transform::truncate::truncate_lines(&stdout, 500);
                let stderr = transform::truncate::truncate_lines(&stderr, 500);

                let mut out = format!("exit_code: {exit_code}\n");
                if !stdout.trim().is_empty() {
                    out.push_str("stdout:\n");
                    out.push_str(&stdout);
                }
                if !stderr.trim().is_empty() {
                    if !stdout.trim().is_empty() {
                        out.push('\n');
                    }
                    out.push_str("stderr:\n");
                    out.push_str(&stderr);
                }

                if exit_code == 0 {
                    make_result(ToolResultStatus::Success, out)
                } else {
                    make_result(ToolResultStatus::Failure, out)
                }
            }
            Ok(Err(e)) => make_result(
                ToolResultStatus::Failure,
                format!("Failed to execute command: {e}"),
            ),
            Err(_) => make_result(
                ToolResultStatus::Failure,
                format!("Command timed out after {timeout_ms}ms"),
            ),
        })
    }
}
