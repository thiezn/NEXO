//! Shared utility types used across domain modules.

/// Cursor-based pagination primitives.
pub mod pagination;
/// UTC timestamp primitives.
pub mod timestamp;

pub use pagination::{PageInfo, PageRequest};
pub use timestamp::Timestamp;
