//! Public record types used by stored context assets.

/// Metadata for a markdown document stored under `PREFILL/`.
pub struct ContextDocument {
    /// The filename relative to the `PREFILL/` directory.
    pub filename: String,
}

/// Metadata for a stored collection of context documents.
pub struct ContextCollection {
    /// Stable identifier for the collection.
    pub id: String,
    /// Human-readable collection name.
    pub name: String,
    /// Optional description shown to callers.
    pub description: Option<String>,
    /// Ordered markdown filenames that make up the collection.
    pub markdown_files: Vec<String>,
}
