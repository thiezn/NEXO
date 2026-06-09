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
pub struct NotesSearchArgs {
    #[schemars(description = "Query to search for in notes")]
    pub query: String,

    #[schemars(description = "case sensitive search, defaults to false")]
    #[serde(default)]
    pub case_sensitive: bool,
}

/// Executes `notes.search` tool against local filesystem
#[derive(Default)]
pub struct NotesSearchTool<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

#[async_trait]
impl<S> Tool for NotesSearchTool<S>
where
    S: NoteStorage + 'static,
{
    type Args = NotesSearchArgs;

    fn name(&self) -> &str {
        "notes.search"
    }

    fn description(&self) -> &str {
        "Searches for notes matching a query."
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let query = args.query;
        let case_sensitive = args.case_sensitive;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform searching in a blocking task to avoid blocking the async runtime
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut matches: Vec<Note> = Vec::new();

            if case_sensitive {
                for filename in storage.list_note_filenames()? {
                    let note = storage.read_note(&filename)?;

                    if note.title.contains(&query)
                        || note
                            .categories
                            .as_ref()
                            .map(|categories| {
                                categories
                                    .iter()
                                    .any(|category| category.title.contains(&query))
                            })
                            .unwrap_or(false)
                        || note.body.contains(&query)
                    {
                        matches.push(note);
                    }
                }
            } else {
                let query = query.to_ascii_lowercase();
                for filename in storage.list_note_filenames()? {
                    let note = storage.read_note(&filename)?;

                    if note.title.to_ascii_lowercase().contains(&query)
                        || note
                            .categories
                            .as_ref()
                            .map(|categories| {
                                categories.iter().any(|category| {
                                    category.title.to_ascii_lowercase().contains(&query)
                                })
                            })
                            .unwrap_or(false)
                        || note.body.to_ascii_lowercase().contains(&query)
                    {
                        matches.push(note);
                    }
                }
            }

            if matches.is_empty() {
                Ok::<String, anyhow::Error>("No matching notes found.".to_string())
            } else {
                Ok(matches
                    .into_iter()
                    .map(|note| format!("# File {}:\n{}", note.filename(), note.content()))
                    .collect::<Vec<_>>()
                    .join("\n\n"))
            }
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.search join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => make_result(ToolResultStatus::Success, output),
            Err(e) => make_result(
                ToolResultStatus::Failure,
                format!("notes.search failed: {e}"),
            ),
        })
    }
}
