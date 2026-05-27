use std::sync::Arc;

use async_trait::async_trait;
use nexo_core::{ToolDefinition, ToolResult};

use crate::NoteStorage;

/// Return all note tools backed by the given storage.
pub fn all_tools(storage: Arc<dyn NoteStorage>) -> Vec<ToolDefinition> {
    vec![
        Arc::new(NotesCreate {
            storage: storage.clone(),
        }),
        Arc::new(NotesList {
            storage: storage.clone(),
        }),
        Arc::new(NotesRead {
            storage: storage.clone(),
        }),
        Arc::new(NotesUpdateSummary { storage }),
    ]
}

// ── notes.create ───────────────────────────────────────────────────────────────

struct NotesCreate {
    storage: Arc<dyn NoteStorage>,
}

#[async_trait]
impl ToolDefinition for NotesCreate {
    fn name(&self) -> &str {
        "notes.create"
    }

    fn description(&self) -> &str {
        "Create a timestamped markdown note. Use this to record observations, decisions, \
         or information worth remembering across conversations."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
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
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled")
            .to_string();
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let storage = self.storage.clone();
        tokio::task::spawn_blocking(move || {
            let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S");
            let filename = format!("{timestamp}.md");
            let full_content = format!("# {title}\n\n{content}");
            storage.write_note(&filename, &full_content)?;
            Ok(ToolResult {
                success: true,
                output: format!("Note created: {filename}"),
                error: None,
            })
        })
        .await?
    }
}

// ── notes.list ─────────────────────────────────────────────────────────────────

struct NotesList {
    storage: Arc<dyn NoteStorage>,
}

#[async_trait]
impl ToolDefinition for NotesList {
    fn name(&self) -> &str {
        "notes.list"
    }

    fn description(&self) -> &str {
        "List all saved notes. Returns filenames sorted chronologically."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let storage = self.storage.clone();
        tokio::task::spawn_blocking(move || {
            let notes = storage.list_notes()?;
            let output = if notes.is_empty() {
                "No notes found.".to_string()
            } else {
                notes.join("\n")
            };
            Ok(ToolResult {
                success: true,
                output,
                error: None,
            })
        })
        .await?
    }
}

// ── notes.read ─────────────────────────────────────────────────────────────────

struct NotesRead {
    storage: Arc<dyn NoteStorage>,
}

#[async_trait]
impl ToolDefinition for NotesRead {
    fn name(&self) -> &str {
        "notes.read"
    }

    fn description(&self) -> &str {
        "Read the contents of a specific note by filename."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the note to read (e.g. 2024-01-01T12-00-00.md)"
                }
            },
            "required": ["filename"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let filename = args
            .get("filename")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: filename"))?
            .to_string();

        let storage = self.storage.clone();
        tokio::task::spawn_blocking(move || {
            let content = storage.read_note(&filename)?;
            Ok(ToolResult {
                success: true,
                output: content,
                error: None,
            })
        })
        .await?
    }
}

// ── notes.update_summary ───────────────────────────────────────────────────────

struct NotesUpdateSummary {
    storage: Arc<dyn NoteStorage>,
}

#[async_trait]
impl ToolDefinition for NotesUpdateSummary {
    fn name(&self) -> &str {
        "notes.update_summary"
    }

    fn description(&self) -> &str {
        "Write or update the notes summary file (NOTES/SUMMARY.md). \
         Use this after reading and organizing all notes into a coherent summary."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Markdown content for the notes summary"
                }
            },
            "required": ["summary"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let summary = args
            .get("summary")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: summary"))?
            .to_string();

        let storage = self.storage.clone();
        tokio::task::spawn_blocking(move || {
            storage.write_summary(&summary)?;
            Ok(ToolResult {
                success: true,
                output: "Summary updated successfully.".to_string(),
                error: None,
            })
        })
        .await?
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

    #[tokio::test]
    async fn create_note_produces_timestamped_file() {
        let storage = Arc::new(MockStorage::new());
        let tools = all_tools(storage.clone());
        let create = tools.iter().find(|t| t.name() == "notes.create").unwrap();

        let result = create
            .execute(serde_json::json!({
                "title": "Test Note",
                "content": "Hello world"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.starts_with("Note created:"));

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

        let tools = all_tools(storage);
        let list = tools.iter().find(|t| t.name() == "notes.list").unwrap();

        let result = list.execute(serde_json::json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("a.md"));
        assert!(result.output.contains("b.md"));
    }

    #[tokio::test]
    async fn read_note_returns_content() {
        let storage = Arc::new(MockStorage::new());
        storage.write_note("test.md", "hello").unwrap();

        let tools = all_tools(storage);
        let read = tools.iter().find(|t| t.name() == "notes.read").unwrap();

        let result = read
            .execute(serde_json::json!({"filename": "test.md"}))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "hello");
    }

    #[tokio::test]
    async fn update_summary_writes_content() {
        let storage = Arc::new(MockStorage::new());
        let tools = all_tools(storage.clone());
        let update = tools
            .iter()
            .find(|t| t.name() == "notes.update_summary")
            .unwrap();

        let result = update
            .execute(serde_json::json!({"summary": "# Summary\n\nAll good."}))
            .await
            .unwrap();
        assert!(result.success);

        let summary = storage.read_summary().unwrap().unwrap();
        assert!(summary.contains("All good."));
    }
}
