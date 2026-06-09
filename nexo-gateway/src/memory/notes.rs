use super::git::GitStorage;
use nexo_notes::{Note, NoteStorage};

/// Bridge `GitStorage` into the `NoteStorage` trait.
impl NoteStorage for GitStorage {
    fn write_note(&self, note: Note) -> anyhow::Result<()> {
        let filename = note.filename();
        self.write_and_sync(
            &format!("NOTES/{filename}"),
            &note.into_content(),
            &format!("Add note: {filename}"),
        )
    }

    fn read_note(&self, filename: &str) -> anyhow::Result<Note> {
        let content = self.read_file(&format!("NOTES/{filename}"))?;
        Ok(Note::from_string(content)?)
    }

    fn list_note_filenames(&self) -> anyhow::Result<Vec<String>> {
        Ok(self
            .list_files("NOTES/")?
            .into_iter()
            .filter(|f| f != "SUMMARY.md")
            .collect())
    }

    fn list_notes(&self) -> anyhow::Result<Vec<Note>> {
        let filenames = self.list_note_filenames()?;
        let mut notes = Vec::new();
        for filename in filenames {
            notes.push(self.read_note(&filename)?);
        }
        Ok(notes)
    }

    fn delete_note(&self, filename: &str) -> anyhow::Result<bool> {
        let path = format!("NOTES/{filename}");
        if !self.file_exists(&path) {
            return Ok(false);
        }
        self.delete_and_sync(&path, &format!("Remove note: {filename}"))?;
        Ok(true)
    }

    fn write_summary(&self, content: &str) -> anyhow::Result<()> {
        self.write_and_sync("NOTES/SUMMARY.md", content, "Update notes summary")
    }

    fn read_summary(&self) -> anyhow::Result<Option<String>> {
        if !self.file_exists("NOTES/SUMMARY.md") {
            return Ok(None);
        }
        Ok(Some(self.read_file("NOTES/SUMMARY.md")?))
    }
}
