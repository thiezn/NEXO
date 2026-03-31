use regex::Regex;
use std::sync::LazyLock;

pub mod ansi;
pub mod code_filter;
pub mod html;
pub mod json;
pub mod truncate;

static MULTIPLE_BLANK_LINES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid blank-lines regex"));

/// Normalize runs of 3+ blank lines down to 2.
pub fn normalize_blank_lines(text: &str) -> String {
    MULTIPLE_BLANK_LINES.replace_all(text, "\n\n").to_string()
}
