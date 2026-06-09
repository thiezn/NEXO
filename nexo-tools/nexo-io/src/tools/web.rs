use crate::transform;
use async_trait::async_trait;
use nexo_core::{
    Result, Tool, ToolCall, ToolExecutionConstraints, ToolParallelism, ToolResult,
    ToolResultContent, ToolResultStatus, ToolSideEffectLevel,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Arguments for the `web_fetch` tool, which fetches a web page or API endpoint.
#[derive(Deserialize, Serialize, JsonSchema)]
pub struct WebFetchArgs {
    /// URL to fetch
    #[schemars(description = "URL to fetch")]
    pub url: String,
}

/// Executes `io.web_fetch` tool against local filesystem
#[derive(Default)]
pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    type Args = WebFetchArgs;

    fn name(&self) -> &str {
        "io.web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a web page or API endpoint. HTML responses are converted to readable markdown. JSON responses are compacted. Plain text is returned as-is."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints {
            side_effect_level: ToolSideEffectLevel::ReadOnly,
            parallelism: ToolParallelism::ParallelGlobal,
            timeout_ms: Some(30_000),
        }
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let url = args.url;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        let response = match reqwest::get(&url).await {
            Ok(r) => r,
            Err(e) => {
                return Ok(make_result(
                    ToolResultStatus::Failure,
                    format!("Request failed: {e}"),
                ));
            }
        };

        let status = response.status();
        if !status.is_success() {
            return Ok(make_result(
                ToolResultStatus::Failure,
                format!("HTTP {status}"),
            ));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let body = match response.text().await {
            Ok(b) => b,
            Err(e) => {
                return Ok(make_result(
                    ToolResultStatus::Failure,
                    format!("Failed to read response body: {e}"),
                ));
            }
        };

        let output = if content_type.contains("text/html") {
            transform::html::html_to_markdown(&body).await
        } else if content_type.contains("application/json") {
            match transform::json::compact_json(&body, 5) {
                Ok(compact) => compact,
                Err(_) => body,
            }
        } else {
            body
        };

        Ok(make_result(ToolResultStatus::Success, output))
    }
}
