//! Note tools for creating, listing, reading, and summarizing markdown notes.

mod storage;
/// Tool definitions and executor implementation for note operations.
pub mod tools;

/// Storage abstraction used by note tool executors.
pub use storage::NoteStorage;
