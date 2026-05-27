//! Shared utility types used across domain modules.

/// Arbitrary JSON metadata helpers.
pub mod metadata;
/// Cursor-based pagination primitives.
pub mod pagination;
/// UTC timestamp primitives.
pub mod timestamp;

pub use metadata::MetadataMap;
pub use pagination::{PageInfo, PageRequest};
pub use timestamp::Timestamp;
