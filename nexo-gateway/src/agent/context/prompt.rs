//! Prompt assembly helpers for agent runs.

use nexo_ws_schema::ToolEntry;

use crate::memory::git::GitStorage;
use crate::server::state::SharedState;

use super::{collections::load_context_collection_index, read_context_document};

/// Prompt assets loaded from storage for a single agent run.
pub struct SystemPromptAssets {
    /// Contents of `SOUL.md`, or an empty string when unavailable.
    pub soul_markdown: String,
    /// Optional concatenated content from the selected stored context collection.
    pub collection_markdown: Option<String>,
}

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

/// Load the stored prompt assets that contribute to a run's system prompt.
pub async fn load_system_prompt_assets(
    state: &SharedState,
    collection_id: Option<&str>,
) -> SystemPromptAssets {
    let git = state.read().await.git_storage.clone();
    if let Some(git) = git {
        let selected_collection_id = collection_id.map(str::to_owned);
        tokio::task::spawn_blocking(move || {
            let soul_markdown = git.read_file("SOUL.md").unwrap_or_default();
            let collection_markdown = selected_collection_id.and_then(|collection_id| {
                load_collection_prompt(&git, &collection_id)
                    .ok()
                    .flatten()
                    .map(|(content, _)| content)
            });
            SystemPromptAssets {
                soul_markdown,
                collection_markdown,
            }
        })
        .await
        .unwrap_or(SystemPromptAssets {
            soul_markdown: String::new(),
            collection_markdown: None,
        })
    } else {
        SystemPromptAssets {
            soul_markdown: String::new(),
            collection_markdown: None,
        }
    }
}

/// Load a stored context collection into concatenated markdown content plus its SHA-256 digest.
///
/// Returns `Ok(None)` when the collection does not exist.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be loaded from storage.
fn load_collection_prompt(
    git: &GitStorage,
    collection_id: &str,
) -> anyhow::Result<Option<(String, String)>> {
    let collections_file = load_context_collection_index(git);
    let collection = match collections_file
        .collections
        .iter()
        .find(|collection| collection.id == collection_id)
    {
        Some(collection) => collection,
        None => return Ok(None),
    };

    let mut parts = Vec::new();
    for filename in &collection.markdown_files {
        match read_context_document(git, filename) {
            Ok(content) => parts.push(content),
            Err(error) => {
                tracing::warn!("Failed to read prefill file '{filename}': {error}");
            }
        }
    }

    let combined = parts.join("\n\n");
    let sha = compute_content_sha(&combined);
    Ok(Some((combined, sha)))
}

/// Compute the SHA-256 hex digest for assembled prompt content.
fn compute_content_sha(content: &str) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(content.as_bytes());
    hex_encode(digest.as_ref())
}

/// Encode raw bytes as lowercase hexadecimal.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
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
