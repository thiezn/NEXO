use async_trait::async_trait;
use nexo_core::{Result, Tool, ToolCall, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Strongly-typed schema parameters mapped exactly to your custom JSON fields
#[derive(Deserialize, Serialize, JsonSchema)]
pub struct SearchArgs {
    /// The file search path/query string
    #[schemars(description = "Bash command to execute")]
    pub query: String,

    /// 0-indexed starting offset or page
    #[schemars(description = "0-indexed starting offset or page")]
    pub page_offset: Option<usize>,

    /// Maximum number of results to return
    #[schemars(description = "Maximum number of results to return")]
    pub max_results: Option<usize>,
}

/// Fuck this seems to be very fast: https://fff.dmtrkovalenko.dev/?repo=1&mode=grep&q=SOCK_RAW
/// pub(crate) async fn execute_grep(call: ToolCall) -> Result<ToolResult> {}
/// Executes `io.search` tool against local filesystem
#[derive(Default)]
pub struct SearchTool;

#[async_trait]
impl Tool for SearchTool {
    type Args = SearchArgs;

    fn name(&self) -> &str {
        "io.search"
    }

    fn description(&self) -> &str {
        "Search for files or content with optional pagination and result limits"
    }

    async fn execute(&self, _call: ToolCall) -> Result<ToolResult> {
        todo!("Implement search and grep");
    }
}
