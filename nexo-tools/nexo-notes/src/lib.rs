//! Note tools for creating, categorizing, listing, reading, and summarizing markdown notes.

mod models;
mod storage;

pub(crate) mod frontmatter;
/// Tool definitions and executor implementation for note operations.
pub mod tools;

pub use tools::NotesCreateTool;
pub use tools::NotesDeleteTool;
pub use tools::NotesEditTool;
pub use tools::NotesListCategoriesTool;
pub use tools::NotesListTool;
pub use tools::NotesReadTool;
pub use tools::NotesSearchTool;
pub use tools::NotesUpdateCategoriesTool;
pub use tools::NotesUpdateSummaryTool;

/// Storage abstraction used by note tool executors.
pub use storage::NoteStorage;

pub use models::Note;
pub use models::NoteCategory;

/// Registers all note tools with the provided `ToolRegistry` and `NoteStorage` implementation.
pub fn register_all_tools<S: NoteStorage + 'static>(
    registry: &mut nexo_core::ToolRegistry,
    storage: std::sync::Arc<S>,
) -> nexo_core::Result<()> {
    registry.register(NotesCreateTool::new(storage.clone()))?;
    registry.register(NotesDeleteTool::new(storage.clone()))?;
    registry.register(NotesEditTool::new(storage.clone()))?;
    registry.register(NotesListCategoriesTool::new(storage.clone()))?;
    registry.register(NotesListTool::new(storage.clone()))?;
    registry.register(NotesReadTool::new(storage.clone()))?;
    registry.register(NotesSearchTool::new(storage.clone()))?;
    registry.register(NotesUpdateCategoriesTool::new(storage.clone()))?;
    registry.register(NotesUpdateSummaryTool::new(storage.clone()))?;
    Ok(())
}
