//! CRUD helpers for stored prompt markdown documents.

use crate::memory::git::GitStorage;
use nexo_spec::prompt::PromptDocument;

const PROMPTS_DIR: &str = "PROMPTS/";

fn prompt_document_path(document_id: &str) -> String {
    format!("{PROMPTS_DIR}{document_id}")
}

/// Create a new stored prompt document and commit it to git.
///
/// # Errors
///
/// Returns an error when the file cannot be written or the git sync fails.
pub fn create_prompt_document(git: &GitStorage, document: &PromptDocument) -> anyhow::Result<()> {
    let path = prompt_document_path(&document.id);
    git.write_and_sync(
        &path,
        &document.content,
        &format!("Add prompt document: {}", document.id),
    )
}

/// List all stored prompt document IDs, excluding `collections.json`.
///
/// # Errors
///
/// Returns an error when the git-backed storage cannot be listed.
pub fn list_prompt_documents(git: &GitStorage) -> anyhow::Result<Vec<String>> {
    let files = git.list_files(PROMPTS_DIR)?;
    Ok(files
        .into_iter()
        .filter(|file| file.ends_with(".md"))
        .collect())
}

/// Read a stored prompt document from `PROMPTS/`.
///
/// # Errors
///
/// Returns an error when the file does not exist or cannot be read.
pub fn read_prompt_document(git: &GitStorage, document_id: &str) -> anyhow::Result<PromptDocument> {
    let content = git.read_file(&prompt_document_path(document_id))?;
    Ok(PromptDocument {
        id: document_id.to_string(),
        content,
    })
}

/// Delete a stored prompt document and commit the change.
///
/// # Errors
///
/// Returns an error when deletion or git sync fails.
pub fn delete_prompt_document(git: &GitStorage, document_id: &str) -> anyhow::Result<bool> {
    let path = prompt_document_path(document_id);
    if !git.file_exists(&path) {
        return Ok(false);
    }
    git.delete_and_sync(&path, &format!("Remove prompt document: {document_id}"))?;
    Ok(true)
}
