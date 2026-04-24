use regex::Regex;
use std::sync::LazyLock;

static ANSI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[a-zA-Z]").expect("valid ANSI regex"));

/// Strip ANSI escape codes (colors, styles) from a string.
pub fn strip_ansi(text: &str) -> String {
    ANSI_RE.replace_all(text, "").into_owned()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn strips_color_codes() {
        assert_eq!(strip_ansi("\x1b[31mError\x1b[0m"), "Error");
    }

    #[test]
    fn strips_multiple_codes() {
        assert_eq!(strip_ansi("\x1b[1m\x1b[32mOK\x1b[0m done"), "OK done");
    }

    #[test]
    fn passthrough_no_codes() {
        assert_eq!(strip_ansi("plain text"), "plain text");
    }

    #[test]
    fn empty_string() {
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn complex_interleaved() {
        let input = "line1\n\x1b[33mwarning:\x1b[0m something\nline3";
        assert_eq!(strip_ansi(input), "line1\nwarning: something\nline3");
    }
}
