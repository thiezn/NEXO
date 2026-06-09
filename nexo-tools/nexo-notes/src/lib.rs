//! Note tools for creating, categorizing, listing, reading, and summarizing markdown notes.

mod models;
mod storage;

pub(crate) mod frontmatter;
/// Tool definitions and executor implementation for note operations.
pub mod tools;

/// Storage abstraction used by note tool executors.
pub use storage::NoteStorage;

pub use models::Note;
pub use models::NoteCategory;
