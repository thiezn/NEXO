/// Abstraction over note storage, decoupling note tools from the git backend.
///
/// All methods are synchronous — callers (tool implementations) wrap them in
/// `tokio::task::spawn_blocking`.
pub trait NoteStorage: Send + Sync {
    /// Write a note file. `filename` is e.g. `"2024-01-01T12-00-00.md"`.
    fn write_note(&self, filename: &str, content: &str) -> anyhow::Result<()>;

    /// Read a note file by filename.
    fn read_note(&self, filename: &str) -> anyhow::Result<String>;

    /// List all note filenames, sorted chronologically (by filename).
    fn list_notes(&self) -> anyhow::Result<Vec<String>>;

    /// Delete a note file. Returns `true` if the file existed.
    fn delete_note(&self, filename: &str) -> anyhow::Result<bool>;

    /// Write or overwrite the summary file (`NOTES/SUMMARY.md`).
    fn write_summary(&self, content: &str) -> anyhow::Result<()>;

    /// Read the summary file, if it exists.
    fn read_summary(&self) -> anyhow::Result<Option<String>>;
}
