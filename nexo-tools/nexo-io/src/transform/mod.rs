//! Text and content normalization helpers shared by `io.*` tools.

use regex::Regex;
use std::sync::LazyLock;

/// ANSI escape-sequence stripping utilities.
pub mod ansi;
/// Language-aware comment filtering for source files.
pub mod code_filter;
/// HTML to markdown conversion utilities.
pub mod html;
/// JSON formatting and compaction helpers.
pub mod json;

static MULTIPLE_BLANK_LINES: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"\n{3,}"));

/// Normalize runs of 3+ blank lines down to 2.
pub fn normalize_blank_lines(text: &str) -> String {
    match &*MULTIPLE_BLANK_LINES {
        Ok(regex) => regex.replace_all(text, "\n\n").to_string(),
        Err(_) => text.to_string(),
    }
}
