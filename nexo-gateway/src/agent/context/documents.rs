//! CRUD helpers for stored context markdown documents.

use crate::memory::git::GitStorage;

use super::ContextDocument;

/// Create a new stored context document and commit it to git.
///
/// # Errors
///
/// Returns an error when the file cannot be written or the git sync fails.
pub fn create_context_document(
    git: &GitStorage,
    filename: &str,
    content: &str,
) -> anyhow::Result<()> {
    let path = format!("PREFILL/{filename}");
    git.write_and_sync(&path, content, &format!("Add prefill: {filename}"))
}

/// List all stored context documents, excluding `collections.json`.
///
/// # Errors
///
/// Returns an error when the git-backed storage cannot be listed.
pub fn list_context_documents(git: &GitStorage) -> anyhow::Result<Vec<ContextDocument>> {
    let files = git.list_files("PREFILL/")?;
    Ok(files
        .into_iter()
        .filter(|file| file.ends_with(".md"))
        .map(|filename| ContextDocument { filename })
        .collect())
}

/// Read a stored context document from `PREFILL/`.
///
/// # Errors
///
/// Returns an error when the file does not exist or cannot be read.
pub fn read_context_document(git: &GitStorage, filename: &str) -> anyhow::Result<String> {
    git.read_file(&format!("PREFILL/{filename}"))
}

/// Delete a stored context document and commit the change.
///
/// # Errors
///
/// Returns an error when deletion or git sync fails.
pub fn delete_context_document(git: &GitStorage, filename: &str) -> anyhow::Result<bool> {
    let path = format!("PREFILL/{filename}");
    if !git.file_exists(&path) {
        return Ok(false);
    }
    git.delete_and_sync(&path, &format!("Remove prefill: {filename}"))?;
    Ok(true)
}
