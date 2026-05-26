//! Prompt assembly helpers for runs.

use nexo_spec::prompt::SystemPrompt;
use nexo_ws_schema::ToolEntry;

use crate::memory::git::GitStorage;
use crate::server::state::SharedState;

use super::{collections::load_prompt_collections, read_prompt_document};

/// Build the tool-instruction section appended to the system prompt.
pub fn build_tool_prompt_section(tools: &[ToolEntry]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out = String::from("# Available Tools\n\n");
    for tool in tools.iter().filter(|tool| tool.available) {
        out.push_str(&format!("## {}\n", tool.spec.name));
        out.push_str(&format!("{}\n", tool.spec.description));
        out.push_str(&format!(
            "Parameters: {}\n",
            serde_json::to_string(&tool.spec.parameters).unwrap_or_default()
        ));
        out.push('\n');
    }
    out
}

/// Load the stored prompt that contributes to a run's system prompt.
pub async fn load_system_prompt(
    state: &SharedState,
    collection_id: Option<&str>,
) -> Option<SystemPrompt> {
    let git = state.read().await.git_storage.clone();
    if let Some(git) = git {
        let selected_collection_id = collection_id.map(str::to_owned);
        tokio::task::spawn_blocking(move || {
            selected_collection_id.and_then(|collection_id| {
                load_prompt_collection_system_prompt(&git, &collection_id)
                    .ok()
                    .flatten()
            })
        })
        .await
        .unwrap_or(None)
    } else {
        None
    }
}

/// Load a stored prompt collection into concatenated markdown content.
/// Returns `Ok(None)` when the collection does not exist.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be loaded from storage.
fn load_prompt_collection_system_prompt(
    git: &GitStorage,
    collection_id: &str,
) -> anyhow::Result<Option<SystemPrompt>> {
    let collections = load_prompt_collections(git);
    let collection = match collections
        .iter()
        .find(|collection| collection.id == collection_id)
    {
        Some(collection) => collection,
        None => return Ok(None),
    };

    let mut parts = Vec::new();
    for document_id in &collection.documents {
        match read_prompt_document(git, document_id) {
            Ok(document) => parts.push(document.content),
            Err(error) => {
                tracing::warn!("Failed to read prompt document '{document_id}': {error}");
            }
        }
    }

    let content = parts.join("\n\n");
    if content.is_empty() {
        return Ok(None);
    }

    Ok(Some(SystemPrompt { content }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tool_prompt_section_empty() {
        let result = build_tool_prompt_section(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn build_tool_prompt_section_formats_available_tools() {
        let tools = vec![
            ToolEntry::new(
                nexo_ws_schema::ToolSpecEntry {
                    name: "echo.run".into(),
                    description: "Echoes input".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: Some("2026-05-22".into()),
                    execution: Default::default(),
                },
                "node",
                true,
            ),
            ToolEntry::new(
                nexo_ws_schema::ToolSpecEntry {
                    name: "offline.tool".into(),
                    description: "Not available".into(),
                    parameters: serde_json::json!({"type": "object"}),
                    contract_version: None,
                    execution: Default::default(),
                },
                "node",
                false,
            ),
        ];
        let result = build_tool_prompt_section(&tools);
        assert!(result.contains("echo.run"));
        assert!(result.contains("Echoes input"));
        assert!(!result.contains("offline.tool"));
    }
}
