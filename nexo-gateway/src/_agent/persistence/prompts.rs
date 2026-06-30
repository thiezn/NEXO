//! Prompt document and collection persistence helpers.

use crate::memory::git::GitStorage;
use nexo_ws_schema::{PromptCollection, PromptDocument, SystemPrompt};
use serde::Deserialize;

const PROMPTS_DIR: &str = "PROMPTS/";
const COLLECTIONS_PATH: &str = "PROMPTS/collections.json";

fn prompt_document_path(document_id: &str) -> String {
    format!("{PROMPTS_DIR}{document_id}")
}

fn deserialize_prompt_collections(content: &str) -> anyhow::Result<Vec<PromptCollection>> {
    if let Ok(collections) = serde_json::from_str::<Vec<PromptCollection>>(content) {
        return Ok(collections);
    }

    #[derive(Debug, Deserialize)]
    struct LegacyPromptCollectionsFile {
        #[serde(default)]
        collections: Vec<PromptCollection>,
    }

    Ok(serde_json::from_str::<LegacyPromptCollectionsFile>(content)?.collections)
}

fn load_prompt_collections(git: &GitStorage) -> Vec<PromptCollection> {
    let content = match git.read_file(COLLECTIONS_PATH) {
        Ok(content) => content,
        Err(_) => return Vec::new(),
    };

    deserialize_prompt_collections(&content).unwrap_or_else(|error| {
        tracing::warn!("Failed to parse prompt collections from '{COLLECTIONS_PATH}': {error}");
        Vec::new()
    })
}

/// Load the concatenated system prompt for a stored collection.
pub(crate) fn load_prompt_collection_system_prompt(
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

/// Create a new stored prompt document and commit it to git.
pub fn create_prompt_document(git: &GitStorage, document: &PromptDocument) -> anyhow::Result<()> {
    let path = prompt_document_path(&document.id);
    git.write_and_sync(
        &path,
        &document.content,
        &format!("Add prompt document: {}", document.id),
    )
}

/// List all stored prompt collections.
pub fn list_prompt_collections(git: &GitStorage) -> anyhow::Result<Vec<PromptCollection>> {
    Ok(load_prompt_collections(git))
}

/// List all stored prompt document IDs, excluding non-markdown files.
pub fn list_prompt_documents(git: &GitStorage) -> anyhow::Result<Vec<String>> {
    let files = git.list_files(PROMPTS_DIR)?;
    Ok(files
        .into_iter()
        .filter(|file| file.ends_with(".md"))
        .collect())
}

/// Read a stored prompt document from `PROMPTS/`.
fn read_prompt_document(git: &GitStorage, document_id: &str) -> anyhow::Result<PromptDocument> {
    let content = git.read_file(&prompt_document_path(document_id))?;
    Ok(PromptDocument {
        id: document_id.to_string(),
        content,
    })
}

/// Create or replace a stored prompt collection with the same ID.
pub fn upsert_prompt_collection(
    git: &GitStorage,
    collection: &PromptCollection,
) -> anyhow::Result<()> {
    let mut collections = load_prompt_collections(git);
    collections.retain(|existing| existing.id != collection.id);
    collections.push(collection.clone());
    git.write_json_and_sync(
        COLLECTIONS_PATH,
        &collections,
        &format!("Add prompt collection: {}", collection.id),
    )
}

/// Delete a stored prompt document and commit the change.
pub fn delete_prompt_document(git: &GitStorage, document_id: &str) -> anyhow::Result<bool> {
    let path = prompt_document_path(document_id);
    if !git.file_exists(&path) {
        return Ok(false);
    }
    git.delete_and_sync(&path, &format!("Remove prompt document: {document_id}"))?;
    Ok(true)
}

/// Delete a stored prompt collection by ID.
pub fn delete_prompt_collection(git: &GitStorage, id: &str) -> anyhow::Result<bool> {
    let mut collections = load_prompt_collections(git);
    let before = collections.len();
    collections.retain(|collection| collection.id != id);
    if collections.len() == before {
        return Ok(false);
    }
    git.write_json_and_sync(
        COLLECTIONS_PATH,
        &collections,
        &format!("Remove prompt collection: {id}"),
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_collection() -> PromptCollection {
        PromptCollection {
            id: "default".into(),
            name: "Default".into(),
            description: Some("Core identity".into()),
            documents: vec!["identity.md".into(), "skills.md".into()],
        }
    }

    #[test]
    fn deserialize_prompt_collections_accepts_flat_array() {
        let json = serde_json::to_string(&vec![sample_collection()]).unwrap_or_default();
        let parsed = deserialize_prompt_collections(&json);

        assert!(parsed.is_ok(), "parse failed: {:?}", parsed.err());
        assert_eq!(parsed.unwrap_or_default(), vec![sample_collection()]);
    }

    #[test]
    fn deserialize_prompt_collections_accepts_legacy_wrapped_shape() {
        let json = r#"{"collections":[{"id":"default","name":"Default","description":"Core identity","documents":["identity.md","skills.md"]}]}"#;
        let parsed = deserialize_prompt_collections(json);

        assert!(parsed.is_ok(), "parse failed: {:?}", parsed.err());
        assert_eq!(parsed.unwrap_or_default(), vec![sample_collection()]);
    }
}
