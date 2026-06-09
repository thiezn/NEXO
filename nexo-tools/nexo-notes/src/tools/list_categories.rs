use crate::NoteStorage;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesListCategoriesArgs {
    #[schemars(description = "Start date for filtering notes (optional)")]
    pub start_date: Option<DateTime<Utc>>,

    #[schemars(description = "End date for filtering notes (optional)")]
    pub end_date: Option<DateTime<Utc>>,
}

/// Executes `notes.list_categories` tool against local filesystem
#[derive(Default)]
pub struct NotesListCategoriesTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

#[async_trait]
impl<S> Tool for NotesListCategoriesTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesListCategoriesArgs;

    fn name(&self) -> &str {
        "notes.list_categories"
    }

    fn description(&self) -> &str {
        "Lists all categories for notes."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let start_date = args.start_date;
        let end_date = args.end_date;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform listing in a blocking task to avoid blocking the async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut categories = HashSet::<String>::new();

            for filename in storage.list_note_filenames()? {
                let note = storage.read_note(&filename)?;

                if let Some(start_date) = start_date
                    && note.datetime < start_date
                {
                    continue;
                }

                if let Some(end_date) = end_date
                    && note.datetime > end_date
                {
                    continue;
                }

                if let Some(note_categories) = note.categories {
                    for category in note_categories {
                        categories.insert(category.title);
                    }
                }
            }

            Ok::<HashSet<String>, anyhow::Error>(categories)
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.list_categories join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => make_result(ToolResultStatus::Success, Vec::from_iter(output).join("\n")),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.list_categories failed: {e}"),
            ),
        })
    }
}
