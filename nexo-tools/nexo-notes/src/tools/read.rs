use crate::{Note, NoteStorage};
use async_trait::async_trait;
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesReadArgs {
    #[schemars(description = "Note filenames to read")]
    pub filenames: Vec<String>,
}

/// Executes `notes.read` tool against local filesystem
#[derive(Default)]
pub struct NotesReadTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

#[async_trait]
impl<S> Tool for NotesReadTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesReadArgs;

    fn name(&self) -> &str {
        "notes.read"
    }

    fn description(&self) -> &str {
        "Reads a note from the specified file path."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let filenames = args.filenames;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform reading in a blocking task to avoid blocking the async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            filenames
                .into_iter()
                .map(|filename| {
                    let note = storage.read_note(&filename)?;
                    Ok::<Note, anyhow::Error>(note)
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .map(|notes| {
                    notes
                        .into_iter()
                        .map(|note| format!("# Title: {}\n{}\n", note.title, note.body))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.read join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => make_result(ToolResultStatus::Success, output),
            Err(e) => make_result(ToolResultStatus::Failure, format!("notes.read failed: {e}")),
        })
    }
}
