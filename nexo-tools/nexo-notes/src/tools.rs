use std::collections::BTreeMap;
use std::sync::Arc;

use nexo_core::{
    Error, ToolCall, ToolDefinition, ToolExecutionConstraints, ToolExecutor, ToolParallelism,
    ToolResult, ToolResultContent, ToolResultStatus, ToolSideEffectLevel,
};

use crate::NoteStorage;

/// Return all note tools backed by the given storage type.
///
/// The storage value is only used for type inference and API symmetry with
/// executor construction.
pub fn all_tools<S>(_storage: Arc<S>) -> Vec<ToolDefinition>
where
    S: NoteStorage + 'static,
{
    vec![
        ToolDefinition {
            name: "notes.create".to_string(),
            description: "Create a timestamped markdown note. Use this to record observations, decisions, or information worth remembering across conversations.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Brief title for the note"
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content of the note"
                    }
                },
                "required": ["title", "content"]
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::SideEffecting,
                parallelism: ToolParallelism::Sequential,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "notes.list".to_string(),
            description: "List all saved notes. Returns filenames sorted chronologically."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "notes.read".to_string(),
            description: "Read the contents of a specific note by filename.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename of the note to read (e.g. 2024-01-01T12-00-00.md)"
                    }
                },
                "required": ["filename"]
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "notes.update_summary".to_string(),
            description: "Write or update the notes summary file (NOTES/SUMMARY.md). Use this after reading and organizing all notes into a coherent summary.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Markdown content for the notes summary"
                    }
                },
                "required": ["summary"]
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::SideEffecting,
                parallelism: ToolParallelism::Sequential,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
    ]
}

/// Executes note tools against the provided storage backend.
pub struct NotesToolExecutor<S>
where
    S: NoteStorage + 'static,
{
    storage: Arc<S>,
}

impl<S> NotesToolExecutor<S>
where
    S: NoteStorage + 'static,
{
    /// Create a new note tool executor backed by the given storage implementation.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend used for all note read/write operations.
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

impl<S> ToolExecutor for NotesToolExecutor<S>
where
    S: NoteStorage + 'static,
{
    async fn execute(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        match call.name.as_str() {
            "notes.create" => self.execute_create(call).await,
            "notes.list" => self.execute_list(call).await,
            "notes.read" => self.execute_read(call).await,
            "notes.update_summary" => self.execute_update_summary(call).await,
            _ => Err(Error::UnsupportedFeature {
                feature: format!("unknown tool: {}", call.name),
            }),
        }
    }
}

impl<S> NotesToolExecutor<S>
where
    S: NoteStorage + 'static,
{
    async fn execute_create(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let title = call
            .arguments
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled")
            .to_string();
        let content = call
            .arguments
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S");
            let filename = format!("{timestamp}.md");
            let full_content = format!("# {title}\n\n{content}");
            storage.write_note(&filename, &full_content)?;
            Ok::<String, anyhow::Error>(format!("Note created: {filename}"))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.create join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => ok_text(call, message),
            Err(e) => fail_text(call, format!("notes.create failed: {e}")),
        })
    }

    async fn execute_list(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let notes = storage.list_notes()?;
            let output = if notes.is_empty() {
                "No notes found.".to_string()
            } else {
                notes.join("\n")
            };
            Ok::<String, anyhow::Error>(output)
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.list join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(e) => fail_text(call, format!("notes.list failed: {e}")),
        })
    }

    async fn execute_read(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let filename = call
            .arguments
            .get("filename")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: filename".to_string(),
            })?
            .to_string();

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || storage.read_note(&filename))
            .await
            .map_err(|e| Error::InvalidState {
                message: format!("notes.read join error: {e}"),
            })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(e) => fail_text(call, format!("notes.read failed: {e}")),
        })
    }

    async fn execute_update_summary(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let summary = call
            .arguments
            .get("summary")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: summary".to_string(),
            })?
            .to_string();

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || storage.write_summary(&summary))
            .await
            .map_err(|e| Error::InvalidState {
                message: format!("notes.update_summary join error: {e}"),
            })?;

        Ok(match outcome {
            Ok(_) => ok_text(call, "Summary updated successfully.".to_string()),
            Err(e) => fail_text(call, format!("notes.update_summary failed: {e}")),
        })
    }
}

fn ok_text(call: ToolCall, output: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id,
        tool_name: call.name,
        status: ToolResultStatus::Success,
        content: ToolResultContent::Text(output),
    }
}

fn fail_text(call: ToolCall, message: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id,
        tool_name: call.name,
        status: ToolResultStatus::Failure,
        content: ToolResultContent::Text(message),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// In-memory mock for testing.
    struct MockStorage {
        notes: Mutex<BTreeMap<String, String>>,
        summary: Mutex<Option<String>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                notes: Mutex::new(BTreeMap::new()),
                summary: Mutex::new(None),
            }
        }
    }

    impl NoteStorage for MockStorage {
        fn write_note(&self, filename: &str, content: &str) -> anyhow::Result<()> {
            self.notes
                .lock()
                .unwrap()
                .insert(filename.to_string(), content.to_string());
            Ok(())
        }

        fn read_note(&self, filename: &str) -> anyhow::Result<String> {
            self.notes
                .lock()
                .unwrap()
                .get(filename)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("not found"))
        }

        fn list_notes(&self) -> anyhow::Result<Vec<String>> {
            Ok(self.notes.lock().unwrap().keys().cloned().collect())
        }

        fn delete_note(&self, filename: &str) -> anyhow::Result<bool> {
            Ok(self.notes.lock().unwrap().remove(filename).is_some())
        }

        fn write_summary(&self, content: &str) -> anyhow::Result<()> {
            *self.summary.lock().unwrap() = Some(content.to_string());
            Ok(())
        }

        fn read_summary(&self) -> anyhow::Result<Option<String>> {
            Ok(self.summary.lock().unwrap().clone())
        }
    }

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "tc-1".into(),
            index: 0,
            name: name.to_string(),
            arguments,
        }
    }

    fn text_content(result: &ToolResult) -> &str {
        match &result.content {
            ToolResultContent::Text(v) => v.as_str(),
            ToolResultContent::Json(_) => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn all_tools_registered() {
        let storage = Arc::new(MockStorage::new());
        let tools = all_tools(storage);
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"notes.create"));
        assert!(names.contains(&"notes.list"));
        assert!(names.contains(&"notes.read"));
        assert!(names.contains(&"notes.update_summary"));
    }

    #[tokio::test]
    async fn create_note_produces_timestamped_file() {
        let storage = Arc::new(MockStorage::new());
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.create",
                serde_json::json!({"title": "Test Note", "content": "Hello world"}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_content(&result).starts_with("Note created:"));

        let notes = storage.list_notes().unwrap();
        assert_eq!(notes.len(), 1);

        let content = storage.read_note(&notes[0]).unwrap();
        assert!(content.contains("# Test Note"));
        assert!(content.contains("Hello world"));
    }

    #[tokio::test]
    async fn list_notes_returns_filenames() {
        let storage = Arc::new(MockStorage::new());
        storage.write_note("a.md", "first").unwrap();
        storage.write_note("b.md", "second").unwrap();

        let executor = NotesToolExecutor::new(storage);
        let result = executor
            .execute(call("notes.list", serde_json::json!({})))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_content(&result).contains("a.md"));
        assert!(text_content(&result).contains("b.md"));
    }

    #[tokio::test]
    async fn read_note_returns_content() {
        let storage = Arc::new(MockStorage::new());
        storage.write_note("test.md", "hello").unwrap();

        let executor = NotesToolExecutor::new(storage);
        let result = executor
            .execute(call(
                "notes.read",
                serde_json::json!({"filename": "test.md"}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(text_content(&result), "hello");
    }

    #[tokio::test]
    async fn update_summary_writes_content() {
        let storage = Arc::new(MockStorage::new());
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.update_summary",
                serde_json::json!({"summary": "# Summary\n\nAll good."}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);

        let summary = storage.read_summary().unwrap().unwrap();
        assert!(summary.contains("All good."));
    }
}
