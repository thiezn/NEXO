//! Collection metadata persistence for stored context assets.

use crate::memory::git::GitStorage;
use serde::{Deserialize, Serialize};

use super::ContextCollection;

const COLLECTIONS_PATH: &str = "PREFILL/collections.json";

/// Serialized representation of `PREFILL/collections.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ContextCollectionsFile {
    #[serde(default)]
    pub(super) collections: Vec<ContextCollectionDef>,
}

/// Serialized representation of a single stored context collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ContextCollectionDef {
    pub(super) id: String,
    pub(super) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<String>,
    #[serde(default)]
    pub(super) markdown_files: Vec<String>,
}

/// Load stored context collection metadata, defaulting to an empty index.
pub(super) fn load_context_collection_index(git: &GitStorage) -> ContextCollectionsFile {
    git.read_json::<ContextCollectionsFile>(COLLECTIONS_PATH)
        .unwrap_or(ContextCollectionsFile {
            collections: Vec::new(),
        })
}

/// Persist the stored context collection index back to git-backed storage.
fn save_context_collection_index(
    git: &GitStorage,
    file: &ContextCollectionsFile,
    commit_message: &str,
) -> anyhow::Result<()> {
    git.write_json_and_sync(COLLECTIONS_PATH, file, commit_message)
}

/// List all stored context collections.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be read from storage.
pub fn list_context_collections(git: &GitStorage) -> anyhow::Result<Vec<ContextCollection>> {
    let file = load_context_collection_index(git);
    Ok(file
        .collections
        .into_iter()
        .map(|collection| ContextCollection {
            id: collection.id,
            name: collection.name,
            description: collection.description,
            markdown_files: collection.markdown_files,
        })
        .collect())
}

/// Create or replace a stored context collection with the same ID.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be written or synced.
pub fn upsert_context_collection(
    git: &GitStorage,
    id: &str,
    name: &str,
    description: Option<&str>,
    files: &[String],
) -> anyhow::Result<()> {
    let mut collections_file = load_context_collection_index(git);
    collections_file
        .collections
        .retain(|collection| collection.id != id);
    collections_file.collections.push(ContextCollectionDef {
        id: id.to_string(),
        name: name.to_string(),
        description: description.map(String::from),
        markdown_files: files.to_vec(),
    });
    save_context_collection_index(git, &collections_file, &format!("Add collection: {id}"))
}

/// Delete a stored context collection by ID.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be written or synced.
pub fn delete_context_collection(git: &GitStorage, id: &str) -> anyhow::Result<bool> {
    let mut collections_file = load_context_collection_index(git);
    let before = collections_file.collections.len();
    collections_file
        .collections
        .retain(|collection| collection.id != id);
    if collections_file.collections.len() == before {
        return Ok(false);
    }
    save_context_collection_index(git, &collections_file, &format!("Remove collection: {id}"))?;
    Ok(true)
}
