use crate::{Note, NoteCategory, NoteStorage};
use async_trait::async_trait;
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct NotesCreateArgs {
    #[schemars(description = "Title of the note to create")]
    pub title: String,

    #[schemars(description = "Body to write to the new note")]
    pub body: String,

    #[schemars(description = "Categories for the new note")]
    pub categories: Option<Vec<NoteCategory>>,
}

/// Executes `notes.create` tool against local filesystem
#[derive(Default)]
pub struct NotesCreateTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

#[async_trait]
impl<S> Tool for NotesCreateTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesCreateArgs;

    fn name(&self) -> &str {
        "notes.create"
    }

    fn description(&self) -> &str {
        "Creates a new note with the specified content at the given file path. If a note already exists at that path, it will be overwritten."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let title = args.title;
        let body = args.body;
        let categories = args.categories;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now();
            let note = Note {
                title,
                datetime: now,
                categories,
                body,
            };
            let filename = note.filename();

            storage.write_note(note)?;
            Ok::<String, anyhow::Error>(format!("Note created: {filename}"))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.create join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => make_result(ToolResultStatus::Success, message),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.create failed: {e}"),
            ),
        })
    }
}
