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
pub struct NotesDeleteArgs {
    #[schemars(description = "Filenames of the notes to delete")]
    pub filenames: Vec<String>,
}

/// Executes `notes.delete` tool against local filesystem
#[derive(Default)]
pub struct NotesDeleteTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesDeleteTool<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new `NotesDeleteTool` with the provided storage implementation.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S> Tool for NotesDeleteTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesDeleteArgs;

    fn name(&self) -> &str {
        "notes.delete"
    }

    fn description(&self) -> &str {
        "Deletes the specified notes."
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

        // Perform deletion in blocking task to avoid blocking async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut deleted_notes = Vec::new();
            let mut missing_notes = Vec::new();
            for filename in filenames {
                if storage.delete_note(&filename)? {
                    deleted_notes.push(filename);
                } else {
                    missing_notes.push(filename);
                }
            }

            Ok::<String, anyhow::Error>(format!(
                "Deleted notes: {:?}\nMissing notes: {:?}",
                deleted_notes, missing_notes
            ))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.delete join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => make_result(ToolResultStatus::Success, message),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.delete failed: {e}"),
            ),
        })
    }
}
