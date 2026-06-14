use crate::NoteStorage;
use async_trait::async_trait;
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesUpdateSummaryArgs {
    #[schemars(description = "Summary to update for the note")]
    pub summary: String,
}

/// Executes `notes.update_summary` tool against local filesystem
#[derive(Default)]
pub struct NotesUpdateSummaryTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesUpdateSummaryTool<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new `NotesUpdateSummaryTool` with the provided storage implementation.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S> Tool for NotesUpdateSummaryTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesUpdateSummaryArgs;

    fn name(&self) -> &str {
        "notes.update_summary"
    }

    fn description(&self) -> &str {
        "Updates the summary for notes."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let summary = args.summary;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || storage.write_summary(&summary))
            .await
            .map_err(|e| Error::InvalidState {
                message: format!("notes.update_summary join error: {e}"),
            })?;

        Ok(match outcome {
            Ok(_) => make_result(
                ToolResultStatus::Success,
                "Summary updated successfully.".to_string(),
            ),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.update_summary failed: {e}"),
            ),
        })
    }
}
