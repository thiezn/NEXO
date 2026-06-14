use crate::{NoteCategory, NoteStorage};
use async_trait::async_trait;
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesUpdateCategoriesArgs {
    #[schemars(description = "Notes to update categories for")]
    pub notes: Vec<NotesSingleUpdateCategoriesArgs>,
}

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesSingleUpdateCategoriesArgs {
    #[schemars(description = "Filename of the note to update categories for")]
    pub filename: String,

    #[schemars(description = "New categories for the note")]
    pub categories: Option<Vec<NoteCategory>>,
}

/// Executes `notes.update_categories` tool against local filesystem
#[derive(Default)]
pub struct NotesUpdateCategoriesTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesUpdateCategoriesTool<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new `NotesUpdateCategoriesTool` with the provided storage implementation.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S> Tool for NotesUpdateCategoriesTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesUpdateCategoriesArgs;

    fn name(&self) -> &str {
        "notes.update_categories"
    }

    fn description(&self) -> &str {
        "Updates categories for a note at the specified file path."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;
        let note_updates = args.notes;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform updating in a blocking task to avoid blocking the async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut updated_notes = 0;
            for update in note_updates {
                let mut note = storage.read_note(&update.filename)?;
                note.categories = update.categories;
                storage.write_note(note)?;
                updated_notes += 1;
            }

            Ok::<String, anyhow::Error>(format!(
                "Categories updated: {updated_notes} notes updated."
            ))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.update_categories join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => make_result(ToolResultStatus::Success, message),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.update_categories failed: {e}"),
            ),
        })
    }
}
