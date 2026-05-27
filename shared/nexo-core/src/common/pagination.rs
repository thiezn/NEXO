use serde::{Deserialize, Serialize};

/// A cursor-based pagination request used by list-oriented APIs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PageRequest {
    /// The opaque cursor returned by a previous page response.
    pub cursor: Option<String>,

    /// The maximum number of items requested for the page.
    pub limit: Option<u32>,
}

/// Pagination metadata returned alongside a page of items.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PageInfo {
    /// The cursor to use for fetching the next page, if one exists.
    pub next_cursor: Option<String>,

    /// Indicates whether additional items are available after the current page.
    pub has_more: bool,
}
