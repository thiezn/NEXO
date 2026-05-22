use crate::memory::git::GitStorage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── Record types ────────────────────────────────────────────────────────────────

pub struct PrefillRecord {
    pub filename: String,
}

pub struct CollectionRecord {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub markdown_files: Vec<String>,
}

// ── JSON schema for PREFILL/collections.json ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionsFile {
    #[serde(default)]
    collections: Vec<CollectionDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionDef {
    id: String,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default)]
    markdown_files: Vec<String>,
}

const COLLECTIONS_PATH: &str = "PREFILL/collections.json";

// ── SHA utility ─────────────────────────────────────────────────────────────────

pub fn compute_sha(combined: &str) -> String {
    let digest = Sha256::digest(combined.as_bytes());
    hex_encode(digest.as_ref())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

// ── Markdown CRUD ───────────────────────────────────────────────────────────────

/// Create a new markdown file in PREFILL/ and commit to git.
pub fn create_markdown(git: &GitStorage, filename: &str, content: &str) -> anyhow::Result<()> {
    let path = format!("PREFILL/{filename}");
    git.write_and_sync(&path, content, &format!("Add prefill: {filename}"))
}

/// List all markdown files in PREFILL/ (excludes collections.json).
pub fn list_markdown(git: &GitStorage) -> anyhow::Result<Vec<PrefillRecord>> {
    let files = git.list_files("PREFILL/")?;
    Ok(files
        .into_iter()
        .filter(|f| f.ends_with(".md"))
        .map(|filename| PrefillRecord { filename })
        .collect())
}

/// Read a markdown file from PREFILL/.
pub fn read_markdown(git: &GitStorage, filename: &str) -> anyhow::Result<String> {
    git.read_file(&format!("PREFILL/{filename}"))
}

/// Delete a markdown file from PREFILL/ and commit. Returns true if it existed.
pub fn delete_markdown(git: &GitStorage, filename: &str) -> anyhow::Result<bool> {
    let path = format!("PREFILL/{filename}");
    if !git.file_exists(&path) {
        return Ok(false);
    }
    git.delete_and_sync(&path, &format!("Remove prefill: {filename}"))?;
    Ok(true)
}

// ── Collection CRUD ─────────────────────────────────────────────────────────────

fn load_collections(git: &GitStorage) -> CollectionsFile {
    git.read_json::<CollectionsFile>(COLLECTIONS_PATH)
        .unwrap_or(CollectionsFile {
            collections: Vec::new(),
        })
}

fn save_collections(git: &GitStorage, file: &CollectionsFile, msg: &str) -> anyhow::Result<()> {
    git.write_json_and_sync(COLLECTIONS_PATH, file, msg)
}

/// List all prefill collections.
pub fn list_collections(git: &GitStorage) -> anyhow::Result<Vec<CollectionRecord>> {
    let file = load_collections(git);
    Ok(file
        .collections
        .into_iter()
        .map(|c| CollectionRecord {
            id: c.id,
            name: c.name,
            description: c.description,
            markdown_files: c.markdown_files,
        })
        .collect())
}

/// Create a new collection (or update if the ID already exists).
pub fn create_collection(
    git: &GitStorage,
    id: &str,
    name: &str,
    description: Option<&str>,
    files: &[String],
) -> anyhow::Result<()> {
    let mut cf = load_collections(git);
    // Remove existing with same ID
    cf.collections.retain(|c| c.id != id);
    cf.collections.push(CollectionDef {
        id: id.to_string(),
        name: name.to_string(),
        description: description.map(String::from),
        markdown_files: files.to_vec(),
    });
    save_collections(git, &cf, &format!("Add collection: {id}"))
}

/// Delete a collection by ID. Returns true if it existed.
pub fn delete_collection(git: &GitStorage, id: &str) -> anyhow::Result<bool> {
    let mut cf = load_collections(git);
    let before = cf.collections.len();
    cf.collections.retain(|c| c.id != id);
    if cf.collections.len() == before {
        return Ok(false);
    }
    save_collections(git, &cf, &format!("Remove collection: {id}"))?;
    Ok(true)
}

// ── Content assembly ────────────────────────────────────────────────────────────

/// Resolve a collection: load each referenced markdown file in order, concatenate,
/// and return `(combined_content, sha256_hex)`. Returns `None` if the collection
/// does not exist.
pub fn resolve_collection(
    git: &GitStorage,
    collection_id: &str,
) -> anyhow::Result<Option<(String, String)>> {
    let cf = load_collections(git);
    let collection = match cf.collections.iter().find(|c| c.id == collection_id) {
        Some(c) => c,
        None => return Ok(None),
    };

    let mut parts: Vec<String> = Vec::new();
    for filename in &collection.markdown_files {
        match read_markdown(git, filename) {
            Ok(content) => parts.push(content),
            Err(e) => {
                tracing::warn!("Failed to read prefill file '{filename}': {e}");
            }
        }
    }

    let combined = parts.join("\n\n");
    let sha = compute_sha(&combined);
    Ok(Some((combined, sha)))
}
