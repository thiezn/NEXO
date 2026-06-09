use crate::transform;
use async_trait::async_trait;
use nexo_core::{Error, Result, Tool, ToolCall, ToolResult, ToolResultContent, ToolResultStatus};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
/// Arguments for the `read` tool, which reads a file with optional offset and limit.
#[derive(Deserialize, Serialize, JsonSchema)]
pub struct ReadArgs {
    /// File path to read
    #[schemars(description = "File path to read")]
    pub path: String,

    /// Offset in lines (default: 0)
    #[schemars(description = "Line number to start reading from (0-indexed)")]
    pub offset: Option<usize>,

    /// Limit in lines (optional)
    #[schemars(description = "Maximum number of lines to return")]
    pub limit: Option<usize>,
}

/// Executes `io.read` tool against local filesystem
#[derive(Default)]
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    type Args = ReadArgs;

    fn name(&self) -> &str {
        "io.read"
    }

    fn description(&self) -> &str {
        "Read a file with optional line offset and limit"
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let path = args.path;
        let offset = args.offset.unwrap_or(0);
        let limit = args.limit;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform the file read in a blocking task to avoid blocking the async runtime.
        let outcome = tokio::task::spawn_blocking(move || {
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => return Err(format!("Failed to read {path}: {e}")),
            };

            let lang = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .map(transform::code_filter::Language::from_extension)
                .unwrap_or(transform::code_filter::Language::Unknown);

            let mut filtered = transform::code_filter::minimal_filter(&content, &lang);
            if filtered.trim().is_empty() && !content.trim().is_empty() {
                filtered = content;
            }

            let lines: Vec<&str> = filtered.lines().collect();
            let start = offset.min(lines.len());
            let end = match limit {
                Some(lim) => (start + lim).min(lines.len()),
                None => lines.len(),
            };

            let sliced = lines[start..end].join("\n");
            Ok(transform::ansi::strip_ansi(&sliced))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("io.read join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => make_result(ToolResultStatus::Success, output),
            Err(message) => make_result(ToolResultStatus::Failure, message),
        })
    }
}
