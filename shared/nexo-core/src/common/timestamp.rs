use chrono::{DateTime, Utc};

/// A UTC timestamp used across shared domain types.
///
/// The wire format follows `chrono`'s serde integration and is emitted as an
/// RFC 3339 timestamp string.
pub type Timestamp = DateTime<Utc>;
