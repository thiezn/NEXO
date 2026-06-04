use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use nexo_core::{
    Error, ToolCall, ToolDefinition, ToolExecutionConstraints, ToolExecutor, ToolParallelism,
    ToolResult, ToolResultContent, ToolResultStatus, ToolSideEffectLevel,
};
use serde::{Deserialize, Serialize};

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
            description: "Create a timestamped markdown note.".to_string(),
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
                    },
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "description": "One or more note categories"
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
            name: "notes.update_categories".to_string(),
            description: "Update note category frontmatter. Try to refine, collapse, or add category names to make categories coherent.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Note filename"
                                },
                                "categories": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "minItems": 1,
                                    "description": "Categories for this note"
                                }
                            },
                            "required": ["filename", "categories"]
                        },
                        "description": "Notes whose category frontmatter should change"
                    }
                },
                "required": ["notes"]
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
            name: "notes.list_categories".to_string(),
            description: "List categories with matching note filenames and titles.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Optional inclusive start date, YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional inclusive end date, YYYY-MM-DD"
                    }
                },
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
                    },
                    "filenames": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Note filenames to read"
                    }
                },
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
            name: "notes.edit".to_string(),
            description: "Edit frontmatter or replace bodies for one or more existing notes.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "The filename of the note to edit"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Replacement frontmatter title"
                                },
                                "date": {
                                    "type": "string",
                                    "description": "Replacement frontmatter date, YYYY-MM-DD"
                                },
                                "time": {
                                    "type": "string",
                                    "description": "Replacement frontmatter time, HH:MM:SSZ"
                                },
                                "categories": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "minItems": 1,
                                    "description": "Replacement frontmatter categories"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Replacement markdown body content"
                                }
                            },
                            "required": ["filename"]
                        },
                        "minItems": 1,
                        "description": "Notes to edit"
                    }
                },
                "required": ["notes"]
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
            name: "notes.search".to_string(),
            description: "Search note titles, categories, and content.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching is case-sensitive"
                    }
                },
                "required": ["query"]
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
            name: "notes.delete".to_string(),
            description: "Delete one or more notes by filename.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "filenames": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "description": "Note filenames to delete"
                    }
                },
                "required": ["filenames"]
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
            "notes.update_categories" => self.execute_update_categories(call).await,
            "notes.list_categories" => self.execute_list_categories(call).await,
            "notes.list" => self.execute_list(call).await,
            "notes.read" => self.execute_read(call).await,
            "notes.edit" => self.execute_edit(call).await,
            "notes.search" => self.execute_search(call).await,
            "notes.delete" => self.execute_delete(call).await,
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
        let categories = optional_categories(&call.arguments, "categories")?.unwrap_or_default();

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now();
            let timestamp = now.format("%Y-%m-%dT%H-%M-%S");
            let filename = format!("{timestamp}.md");
            let metadata = NoteMetadata {
                title: title.clone(),
                date: now.format("%Y-%m-%d").to_string(),
                time: now.format("%H:%M:%SZ").to_string(),
                categories,
            };
            let full_content = render_note_document(&metadata, &format!("# {title}\n\n{content}"))?;
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

    async fn execute_update_categories(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let note_updates = required_note_updates(&call.arguments)?;

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let notes = storage
                .list_notes()?
                .into_iter()
                .map(|filename| {
                    let content = storage.read_note(&filename)?;
                    let parsed = parse_note_document(&filename, &content, chrono::Utc::now())?;
                    Ok::<(String, ParsedNote), anyhow::Error>((filename, parsed))
                })
                .collect::<anyhow::Result<BTreeMap<_, _>>>()?;

            validate_note_updates(&notes, &note_updates)?;

            let mut updated_notes = 0;
            for update in note_updates {
                let Some(note) = notes.get(&update.filename) else {
                    anyhow::bail!("note not found: {}", update.filename);
                };
                let mut metadata = note.metadata.clone();
                metadata.categories = update.categories;
                storage.write_note(
                    &update.filename,
                    &render_note_document(&metadata, &note.body)?,
                )?;
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
            Ok(message) => ok_text(call, message),
            Err(e) => fail_text(call, format!("notes.update_categories failed: {e}")),
        })
    }

    async fn execute_list_categories(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let start_date = optional_date(&call.arguments, "start_date")?;
        let end_date = optional_date(&call.arguments, "end_date")?;

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut categories = BTreeMap::<String, Vec<CategoryNote>>::new();
            for filename in storage.list_notes()? {
                let content = storage.read_note(&filename)?;
                let note = parse_note_document(&filename, &content, chrono::Utc::now())?;
                let note_date = chrono::NaiveDate::parse_from_str(&note.metadata.date, "%Y-%m-%d")?;
                if !date_in_range(note_date, start_date, end_date) {
                    continue;
                }
                for category in &note.metadata.categories {
                    categories
                        .entry(category.clone())
                        .or_default()
                        .push(CategoryNote {
                            filename: filename.clone(),
                            title: note.metadata.title.clone(),
                        });
                }
            }

            Ok::<String, anyhow::Error>(render_categories_output(categories))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.list_categories join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(e) => fail_text(call, format!("notes.list_categories failed: {e}")),
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
        let filenames = requested_filenames(&call.arguments)?;

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            filenames
                .into_iter()
                .map(|filename| {
                    let content = storage.read_note(&filename)?;
                    Ok::<(String, String), anyhow::Error>((filename, content))
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .map(render_notes_output)
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.read join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(e) => fail_text(call, format!("notes.read failed: {e}")),
        })
    }

    async fn execute_edit(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let edits = required_note_edits(&call.arguments)?;

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut edited_notes = 0;
            for edit in edits {
                let existing = storage.read_note(&edit.filename)?;
                let mut note = parse_note_document(&edit.filename, &existing, chrono::Utc::now())?;
                if let Some(title) = edit.title {
                    note.metadata.title = title;
                }
                if let Some(date) = edit.date {
                    validate_date(&date)?;
                    note.metadata.date = date;
                }
                if let Some(time) = edit.time {
                    validate_time(&time)?;
                    note.metadata.time = time;
                }
                if let Some(categories) = edit.categories {
                    note.metadata.categories = categories;
                }
                if let Some(content) = edit.content {
                    note.body = content;
                }

                storage.write_note(
                    &edit.filename,
                    &render_note_document(&note.metadata, &note.body)?,
                )?;
                edited_notes += 1;
            }

            Ok::<String, anyhow::Error>(format!("Notes edited: {edited_notes}"))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.edit join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => ok_text(call, message),
            Err(e) => fail_text(call, format!("notes.edit failed: {e}")),
        })
    }

    async fn execute_delete(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let filenames = required_filenames_array(&call.arguments, "filenames")?;

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

            Ok::<String, anyhow::Error>(render_delete_output(deleted_notes, missing_notes))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.delete join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(message) => ok_text(call, message),
            Err(e) => fail_text(call, format!("notes.delete failed: {e}")),
        })
    }

    async fn execute_search(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let query = required_string(&call.arguments, "query")?;
        let case_sensitive = call
            .arguments
            .get("case_sensitive")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let storage = self.storage.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut matches = Vec::new();
            for filename in storage.list_notes()? {
                let content = storage.read_note(&filename)?;
                let note = parse_note_document(&filename, &content, chrono::Utc::now())?;
                if note_matches(&note, &content, &query, case_sensitive) {
                    matches.push((filename, content));
                }
            }

            if matches.is_empty() {
                Ok::<String, anyhow::Error>("No matching notes found.".to_string())
            } else {
                Ok(render_notes_output(matches))
            }
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("notes.search join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(e) => fail_text(call, format!("notes.search failed: {e}")),
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

#[derive(Clone, Debug, Serialize)]
struct NoteMetadata {
    title: String,
    date: String,
    time: String,
    categories: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawNoteMetadata {
    title: Option<String>,
    date: Option<String>,
    time: Option<String>,
    categories: Option<Vec<String>>,
}

#[derive(Debug)]
struct ParsedNote {
    metadata: NoteMetadata,
    body: String,
}

#[derive(Debug)]
struct NoteCategoryUpdate {
    filename: String,
    categories: Vec<String>,
}

#[derive(Debug)]
struct NoteEdit {
    filename: String,
    title: Option<String>,
    date: Option<String>,
    time: Option<String>,
    categories: Option<Vec<String>>,
    content: Option<String>,
}

#[derive(Debug)]
struct CategoryNote {
    filename: String,
    title: String,
}

fn required_string(arguments: &serde_json::Value, parameter: &str) -> nexo_core::Result<String> {
    arguments
        .get(parameter)
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .ok_or_else(|| Error::InvalidRequest {
            message: format!("missing required parameter: {parameter}"),
        })
}

fn optional_string(
    arguments: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Option<String>> {
    match arguments.get(parameter) {
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_string()))
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("{parameter} must be a string"),
            }),
        None => Ok(None),
    }
}

fn requested_filenames(arguments: &serde_json::Value) -> nexo_core::Result<Vec<String>> {
    let mut filenames = Vec::new();
    if let Some(filename) = arguments.get("filename").and_then(|value| value.as_str()) {
        filenames.push(filename.to_string());
    }
    if let Some(values) = arguments.get("filenames") {
        let values = values.as_array().ok_or_else(|| Error::InvalidRequest {
            message: "filenames must be an array".to_string(),
        })?;
        for value in values {
            let filename = value.as_str().ok_or_else(|| Error::InvalidRequest {
                message: "filenames entries must be strings".to_string(),
            })?;
            filenames.push(filename.to_string());
        }
    }
    let filenames: Vec<String> = filenames
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if filenames.is_empty() {
        return Err(Error::InvalidRequest {
            message: "missing required parameter: filename or filenames".to_string(),
        });
    }
    Ok(filenames)
}

fn required_filenames_array(
    arguments: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Vec<String>> {
    let values = arguments
        .get(parameter)
        .and_then(|value| value.as_array())
        .ok_or_else(|| Error::InvalidRequest {
            message: format!("missing required parameter: {parameter}"),
        })?;

    let filenames = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|filename| !filename.is_empty())
                .map(ToString::to_string)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!("{parameter} entries must be non-empty strings"),
                })
        })
        .collect::<nexo_core::Result<Vec<_>>>()?;
    let filenames: Vec<String> = filenames
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if filenames.is_empty() {
        return Err(Error::InvalidRequest {
            message: format!("{parameter} must include at least one filename"),
        });
    }
    Ok(filenames)
}

fn required_categories(
    arguments: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Vec<String>> {
    let categories = arguments
        .get(parameter)
        .ok_or_else(|| Error::InvalidRequest {
            message: format!("missing required parameter: {parameter}"),
        })
        .and_then(|value| categories_from_value(value, parameter))?;
    if categories.is_empty() {
        return Err(Error::InvalidRequest {
            message: format!("{parameter} must include at least one category"),
        });
    }
    Ok(categories)
}

fn optional_categories(
    arguments: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Option<Vec<String>>> {
    arguments
        .get(parameter)
        .map(|value| categories_from_value(value, parameter))
        .transpose()
}

fn categories_from_value(
    value: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Vec<String>> {
    let values = value.as_array().ok_or_else(|| Error::InvalidRequest {
        message: format!("{parameter} must be an array"),
    })?;
    let categories = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("{parameter} entries must be strings"))
                .and_then(normalize_category)
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .map_err(|e| Error::InvalidRequest {
            message: e.to_string(),
        })?;
    Ok(dedupe_categories(categories))
}

fn required_note_updates(
    arguments: &serde_json::Value,
) -> nexo_core::Result<Vec<NoteCategoryUpdate>> {
    let values = arguments
        .get("notes")
        .and_then(|value| value.as_array())
        .ok_or_else(|| Error::InvalidRequest {
            message: "missing required parameter: notes".to_string(),
        })?;

    values
        .iter()
        .map(|value| {
            let filename = required_string(value, "filename")?;
            let categories = required_categories(value, "categories")?;
            Ok(NoteCategoryUpdate {
                filename,
                categories,
            })
        })
        .collect()
}

fn required_note_edits(arguments: &serde_json::Value) -> nexo_core::Result<Vec<NoteEdit>> {
    let values = arguments
        .get("notes")
        .and_then(|value| value.as_array())
        .ok_or_else(|| Error::InvalidRequest {
            message: "missing required parameter: notes".to_string(),
        })?;

    let edits = values
        .iter()
        .map(|value| {
            Ok(NoteEdit {
                filename: required_string(value, "filename")?,
                title: optional_string(value, "title")?,
                date: optional_string(value, "date")?,
                time: optional_string(value, "time")?,
                categories: optional_categories(value, "categories")?,
                content: optional_string(value, "content")?,
            })
        })
        .collect::<nexo_core::Result<Vec<_>>>()?;

    if edits.is_empty() {
        return Err(Error::InvalidRequest {
            message: "notes must include at least one note edit".to_string(),
        });
    }
    Ok(edits)
}

fn normalize_category(category: &str) -> anyhow::Result<String> {
    let category = category.trim().to_ascii_lowercase();
    if category.is_empty() {
        anyhow::bail!("category cannot be empty");
    }
    if !category
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == ' ' || c == '_' || c == '-')
    {
        anyhow::bail!("category contains unsupported characters: {category}");
    }
    Ok(category)
}

fn dedupe_categories(categories: Vec<String>) -> Vec<String> {
    categories
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn parse_note_document(
    filename: &str,
    content: &str,
    fallback: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<ParsedNote> {
    if let Some((frontmatter, body)) = split_frontmatter(content) {
        let mut metadata = fallback_metadata(filename, body, fallback);
        apply_frontmatter(&mut metadata, yaml_serde::from_str(frontmatter)?)?;
        return Ok(ParsedNote {
            metadata,
            body: body.to_string(),
        });
    }

    Ok(ParsedNote {
        metadata: fallback_metadata(filename, content, fallback),
        body: content.to_string(),
    })
}

fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let rest = content.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let frontmatter = &rest[..end];
    let body = &rest[end + "\n---".len()..];
    Some((frontmatter, body.strip_prefix('\n').unwrap_or(body)))
}

fn fallback_metadata(
    filename: &str,
    content: &str,
    fallback: chrono::DateTime<chrono::Utc>,
) -> NoteMetadata {
    let (date, time) = note_datetime_from_filename(filename, fallback);
    NoteMetadata {
        title: note_title_from_content(content)
            .unwrap_or_else(|| filename.trim_end_matches(".md"))
            .to_string(),
        date,
        time,
        categories: vec!["uncategorized".to_string()],
    }
}

fn apply_frontmatter(
    metadata: &mut NoteMetadata,
    frontmatter: RawNoteMetadata,
) -> anyhow::Result<()> {
    if let Some(title) = frontmatter.title {
        metadata.title = title;
    }
    if let Some(date) = frontmatter.date {
        metadata.date = date;
    }
    if let Some(time) = frontmatter.time {
        metadata.time = time;
    }
    if let Some(categories) = frontmatter.categories {
        metadata.categories = categories
            .iter()
            .map(|category| normalize_category(category))
            .collect::<anyhow::Result<Vec<_>>>()?;
        metadata.categories = dedupe_categories(std::mem::take(&mut metadata.categories));
    }
    Ok(())
}

fn render_note_document(metadata: &NoteMetadata, body: &str) -> anyhow::Result<String> {
    let frontmatter = yaml_serde::to_string(metadata)?;
    let separator = if frontmatter.ends_with('\n') {
        ""
    } else {
        "\n"
    };
    Ok(format!("---\n{frontmatter}{separator}---\n{body}"))
}

fn validate_note_updates(
    notes: &BTreeMap<String, ParsedNote>,
    note_updates: &[NoteCategoryUpdate],
) -> anyhow::Result<()> {
    for update in note_updates {
        if !notes.contains_key(&update.filename) {
            anyhow::bail!("note not found: {}", update.filename);
        }
    }
    Ok(())
}

fn optional_date(
    arguments: &serde_json::Value,
    parameter: &str,
) -> nexo_core::Result<Option<chrono::NaiveDate>> {
    optional_string(arguments, parameter)?
        .map(|value| {
            chrono::NaiveDate::parse_from_str(&value, "%Y-%m-%d").map_err(|e| {
                Error::InvalidRequest {
                    message: format!("{parameter} must be YYYY-MM-DD: {e}"),
                }
            })
        })
        .transpose()
}

fn date_in_range(
    date: chrono::NaiveDate,
    start_date: Option<chrono::NaiveDate>,
    end_date: Option<chrono::NaiveDate>,
) -> bool {
    if let Some(start_date) = start_date
        && date < start_date
    {
        return false;
    }
    if let Some(end_date) = end_date
        && date > end_date
    {
        return false;
    }
    true
}

fn render_categories_output(categories: BTreeMap<String, Vec<CategoryNote>>) -> String {
    if categories.is_empty() {
        return "No categories found.".to_string();
    }

    categories
        .into_iter()
        .map(|(category, mut notes)| {
            notes.sort_by(|left, right| left.filename.cmp(&right.filename));
            let notes = notes
                .into_iter()
                .map(|note| format!("- {} - {}", note.filename, note.title))
                .collect::<Vec<_>>()
                .join("\n");
            format!("## {category}\n\n{notes}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_notes_output(mut notes: Vec<(String, String)>) -> String {
    notes.sort_by(|left, right| left.0.cmp(&right.0));
    notes
        .into_iter()
        .map(|(filename, content)| format!("## {filename}\n\n{content}"))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

fn render_delete_output(deleted_notes: Vec<String>, missing_notes: Vec<String>) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Deleted notes: {}", deleted_notes.len()));
    if !deleted_notes.is_empty() {
        lines.push(deleted_notes.join("\n"));
    }
    if !missing_notes.is_empty() {
        lines.push(format!("Missing notes: {}", missing_notes.len()));
        lines.push(missing_notes.join("\n"));
    }
    lines.join("\n")
}

fn note_matches(note: &ParsedNote, content: &str, query: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        return note.metadata.title.contains(query)
            || note
                .metadata
                .categories
                .iter()
                .any(|category| category.contains(query))
            || content.contains(query);
    }

    let query = query.to_ascii_lowercase();
    note.metadata.title.to_ascii_lowercase().contains(&query)
        || note
            .metadata
            .categories
            .iter()
            .any(|category| category.to_ascii_lowercase().contains(&query))
        || content.to_ascii_lowercase().contains(&query)
}

fn validate_date(date: &str) -> anyhow::Result<()> {
    chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    Ok(())
}

fn validate_time(time: &str) -> anyhow::Result<()> {
    chrono::NaiveTime::parse_from_str(time, "%H:%M:%SZ")?;
    Ok(())
}

fn note_datetime_from_filename(
    filename: &str,
    fallback: chrono::DateTime<chrono::Utc>,
) -> (String, String) {
    match (
        filename.get(0..10),
        filename.get(11..13),
        filename.get(14..16),
        filename.get(17..19),
    ) {
        (Some(date), Some(hour), Some(minute), Some(second)) => {
            (date.to_string(), format!("{hour}:{minute}:{second}Z"))
        }
        _ => (
            fallback.format("%Y-%m-%d").to_string(),
            fallback.format("%H:%M:%SZ").to_string(),
        ),
    }
}

fn note_title_from_content(content: &str) -> Option<&str> {
    content
        .lines()
        .find_map(|line| line.strip_prefix("# ").map(str::trim))
        .filter(|title| !title.is_empty())
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
        assert_eq!(tools.len(), 9);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"notes.create"));
        assert!(names.contains(&"notes.update_categories"));
        assert!(names.contains(&"notes.list_categories"));
        assert!(names.contains(&"notes.list"));
        assert!(names.contains(&"notes.read"));
        assert!(names.contains(&"notes.edit"));
        assert!(names.contains(&"notes.search"));
        assert!(names.contains(&"notes.delete"));
        assert!(names.contains(&"notes.update_summary"));
        assert!(names.iter().all(|name| name.starts_with("notes.")));
    }

    #[tokio::test]
    async fn create_note_produces_timestamped_file() {
        let storage = Arc::new(MockStorage::new());
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.create",
                serde_json::json!({
                    "title": "Test Note",
                    "content": "Hello world",
                    "categories": [" Work ", "work", "rust-stuff"]
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_content(&result).starts_with("Note created:"));

        let notes = storage.list_notes().unwrap();
        assert_eq!(notes.len(), 1);

        let content = storage.read_note(&notes[0]).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("title: Test Note"));
        assert!(content.contains("date: "));
        assert!(content.contains("time: "));
        assert_eq!(content.matches("- work").count(), 1);
        assert!(content.contains("- rust-stuff"));
        assert!(content.contains("# Test Note"));
        assert!(content.contains("Hello world"));
    }

    #[tokio::test]
    async fn create_note_allows_categories_to_be_omitted() {
        let storage = Arc::new(MockStorage::new());
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.create",
                serde_json::json!({"title": "No Category", "content": "Body"}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let notes = storage.list_notes().unwrap();
        let content = storage.read_note(&notes[0]).unwrap();
        assert!(content.contains("categories: []"));
    }

    #[tokio::test]
    async fn update_categories_updates_note_frontmatter() {
        let storage = Arc::new(MockStorage::new());
        storage
            .write_note(
                "2026-06-04T12-00-00.md",
                "---\ntitle: \"Old\"\ndate: 2026-06-04\ntime: 12:00:00Z\ncategories:\n  - old\n---\n# Old\n\nBody",
            )
            .unwrap();
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.update_categories",
                serde_json::json!({
                    "notes": [
                        {"filename": "2026-06-04T12-00-00.md", "categories": [" new "]}
                    ]
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let note = storage.read_note("2026-06-04T12-00-00.md").unwrap();
        assert!(note.contains("categories:\n- new\n---"));
        assert!(!note.contains("  - old"));
    }

    #[tokio::test]
    async fn list_categories_returns_category_notes_with_optional_date_range() {
        let storage = Arc::new(MockStorage::new());
        storage
            .write_note(
                "2026-06-03T12-00-00.md",
                "---\ntitle: Old\ndate: 2026-06-03\ntime: 12:00:00Z\ncategories:\n- rust\n---\n# Old\n\nBody",
            )
            .unwrap();
        storage
            .write_note(
                "2026-06-04T12-00-00.md",
                "---\ntitle: New\ndate: 2026-06-04\ntime: 12:00:00Z\ncategories:\n- rust\n- work\n---\n# New\n\nBody",
            )
            .unwrap();
        let executor = NotesToolExecutor::new(storage);

        let result = executor
            .execute(call(
                "notes.list_categories",
                serde_json::json!({"start_date": "2026-06-04", "end_date": "2026-06-04"}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let output = text_content(&result);
        assert!(output.contains("## rust"));
        assert!(output.contains("- 2026-06-04T12-00-00.md - New"));
        assert!(output.contains("## work"));
        assert!(!output.contains("2026-06-03T12-00-00.md"));
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
    async fn read_note_returns_filename_and_content() {
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
        assert_eq!(text_content(&result), "## test.md\n\nhello");
    }

    #[tokio::test]
    async fn read_notes_returns_multiple_files_sorted_by_date() {
        let storage = Arc::new(MockStorage::new());
        storage
            .write_note("2026-06-05T12-00-00.md", "second")
            .unwrap();
        storage
            .write_note("2026-06-04T12-00-00.md", "first")
            .unwrap();

        let executor = NotesToolExecutor::new(storage);
        let result = executor
            .execute(call(
                "notes.read",
                serde_json::json!({
                    "filenames": ["2026-06-05T12-00-00.md", "2026-06-04T12-00-00.md"]
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let output = text_content(&result);
        let first = output.find("2026-06-04T12-00-00.md").unwrap();
        let second = output.find("2026-06-05T12-00-00.md").unwrap();
        assert!(first < second);
        assert!(output.contains("first"));
        assert!(output.contains("second"));
    }

    #[tokio::test]
    async fn edit_note_updates_frontmatter_and_replaces_body() {
        let storage = Arc::new(MockStorage::new());
        storage
            .write_note(
                "2026-06-04T12-00-00.md",
                "---\ntitle: \"Old\"\ndate: 2026-06-04\ntime: 12:00:00Z\ncategories:\n  - old\n---\n# Old\n\nBody",
            )
            .unwrap();
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.edit",
                serde_json::json!({
                    "notes": [
                        {
                            "filename": "2026-06-04T12-00-00.md",
                            "title": "New",
                            "categories": [" Project "],
                            "content": "# New\n\nReplacement"
                        }
                    ]
                }),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let note = storage.read_note("2026-06-04T12-00-00.md").unwrap();
        assert!(note.contains("title: New"));
        assert!(note.contains("- project"));
        assert!(note.ends_with("# New\n\nReplacement"));
    }

    #[tokio::test]
    async fn delete_notes_deletes_multiple_files() {
        let storage = Arc::new(MockStorage::new());
        storage.write_note("a.md", "first").unwrap();
        storage.write_note("b.md", "second").unwrap();
        let executor = NotesToolExecutor::new(storage.clone());

        let result = executor
            .execute(call(
                "notes.delete",
                serde_json::json!({"filenames": ["b.md", "missing.md", "a.md"]}),
            ))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(storage.list_notes().unwrap(), Vec::<String>::new());
        let output = text_content(&result);
        assert!(output.contains("Deleted notes: 2"));
        assert!(output.contains("Missing notes: 1"));
        assert!(output.contains("missing.md"));
    }

    #[tokio::test]
    async fn search_matches_title_categories_and_content_case_insensitively_by_default() {
        let storage = Arc::new(MockStorage::new());
        storage
            .write_note(
                "2026-06-04T12-00-00.md",
                "---\ntitle: \"Rust Note\"\ndate: 2026-06-04\ntime: 12:00:00Z\ncategories:\n  - language\n---\n# Rust Note\n\nBody",
            )
            .unwrap();
        storage
            .write_note(
                "2026-06-05T12-00-00.md",
                "---\ntitle: \"Other\"\ndate: 2026-06-05\ntime: 12:00:00Z\ncategories:\n  - personal\n---\n# Other\n\nNothing",
            )
            .unwrap();

        let executor = NotesToolExecutor::new(storage);
        let result = executor
            .execute(call("notes.search", serde_json::json!({"query": "rust"})))
            .await
            .unwrap();

        assert_eq!(result.status, ToolResultStatus::Success);
        let output = text_content(&result);
        assert!(output.contains("2026-06-04T12-00-00.md"));
        assert!(output.contains("Rust Note"));
        assert!(!output.contains("2026-06-05T12-00-00.md"));
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
