use crate::{NoteCategory, NoteStorage};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesEditArgs {
    #[schemars(description = "List of notes to edit")]
    pub notes: Vec<NotesSingleEditArgs>,
}

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesSingleEditArgs {
    #[schemars(description = "Filename of the note to edit")]
    pub filename: String,

    #[schemars(description = "New title for the note (optional)")]
    pub title: Option<String>,

    #[schemars(description = "New datetime for the note (optional)")]
    pub datetime: Option<DateTime<Utc>>,

    #[schemars(description = "New categories for the note (optional)")]
    pub categories: Option<Vec<NoteCategory>>,

    #[schemars(description = "New content for the note (optional)")]
    pub content: Option<String>,
}

/// Executes `notes.edit` tool against local filesystem
#[derive(Default)]
pub struct NotesEditTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesEditTool<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new `NotesEditTool` with the provided storage implementation.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S> Tool for NotesEditTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesEditArgs;

    fn name(&self) -> &str {
        "notes.edit"
    }

    fn description(&self) -> &str {
        "Edits the specified note at the given file path."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform edits in a blocking task to avoid blocking the async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut edited_notes = 0;
            for edit in args.notes {
                let mut existing = storage.read_note(&edit.filename)?;
                if let Some(title) = edit.title {
                    existing.title = title;
                }
                if let Some(datetime) = edit.datetime {
                    existing.datetime = datetime;
                }
                if let Some(categories) = edit.categories {
                    existing.categories = Some(categories);
                }
                if let Some(content) = edit.content {
                    existing.body = content;
                }

                storage.write_note(existing)?;
                edited_notes += 1;
            }
            Ok::<String, anyhow::Error>(format!("Notes edited: {edited_notes}"))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.edit join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => make_result(ToolResultStatus::Success, message),
            Err(e) => make_result(ToolResultStatus::Failure, format!("notes.edit failed: {e}")),
        })
    }
}
