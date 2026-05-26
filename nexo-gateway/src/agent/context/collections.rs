//! Collection metadata persistence for stored prompt assets.

use crate::memory::git::GitStorage;
use nexo_spec::prompt::PromptCollection;
use serde::Deserialize;

const COLLECTIONS_PATH: &str = "PROMPTS/collections.json";

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

/// Load stored prompt collections, defaulting to an empty list.
pub(super) fn load_prompt_collections(git: &GitStorage) -> Vec<PromptCollection> {
    let content = match git.read_file(COLLECTIONS_PATH) {
        Ok(content) => content,
        Err(_) => return Vec::new(),
    };

    deserialize_prompt_collections(&content).unwrap_or_else(|error| {
        tracing::warn!("Failed to parse prompt collections from '{COLLECTIONS_PATH}': {error}");
        Vec::new()
    })
}

/// List all stored prompt collections.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be read from storage.
pub fn list_prompt_collections(git: &GitStorage) -> anyhow::Result<Vec<PromptCollection>> {
    Ok(load_prompt_collections(git))
}

/// Create or replace a stored prompt collection with the same ID.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be written or synced.
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

/// Delete a stored prompt collection by ID.
///
/// # Errors
///
/// Returns an error when collection metadata cannot be written or synced.
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
