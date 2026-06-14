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
pub struct NotesListArgs;

/// Executes `notes.list` tool against local filesystem
#[derive(Default)]
pub struct NotesListTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesListTool<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new `NotesListTool` with the provided storage implementation.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S> Tool for NotesListTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesListArgs;

    fn name(&self) -> &str {
        "notes.list"
    }

    fn description(&self) -> &str {
        "Lists all notes at the specified file path."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let notes = storage.list_notes()?;
            let output = if notes.is_empty() {
                "No notes found.".to_string()
            } else {
                notes
                    .into_iter()
                    .map(|note| {
                        format!("# Title: {}\n{}\n", note.title.clone(), note.into_content())
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            };
            Ok::<String, anyhow::Error>(output)
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.list join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => make_result(ToolResultStatus::Success, output),
            Err(e) => make_result(ToolResultStatus::Failure, format!("notes.list failed: {e}")),
        })
    }
}
