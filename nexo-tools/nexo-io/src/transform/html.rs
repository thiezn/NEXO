use regex::Regex;
use std::sync::LazyLock;

static SCRIPT_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>"));
static STYLE_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"(?is)<style[^>]*>.*?</style>"));
static HTML_TAG_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"<[^>]+>"));

/// Convert HTML to readable markdown using async streaming.
///
/// Uses `html2md::rewrite_html_streaming` for tokio-compatible async conversion.
/// Falls back to regex-based tag stripping if conversion produces empty output.
pub async fn html_to_markdown(html: &str) -> String {
    if html.trim().is_empty() {
        return String::new();
    }

    let md = html2md::rewrite_html_streaming(html, false).await;
    let md = post_process(&md);

    // Safety fallback: if conversion emptied non-empty input, use tag stripping
    if md.trim().is_empty() {
        return fallback_strip_tags(html);
    }

    md
}

// /// Synchronous HTML to markdown conversion.
// ///
// /// Uses `html2md::rewrite_html` for use in `spawn_blocking` contexts.
// pub fn html_to_markdown_sync(html: &str) -> String {
//     if html.trim().is_empty() {
//         return String::new();
//     }

//     let md = html2md::rewrite_html(html, false);
//     let md = post_process(&md);

//     if md.trim().is_empty() {
//         return fallback_strip_tags(html);
//     }

//     md
// }

/// Normalize excessive blank lines and trim.
fn post_process(text: &str) -> String {
    super::normalize_blank_lines(text).trim().to_string()
}

/// Fallback: strip HTML tags and decode common entities.
fn fallback_strip_tags(html: &str) -> String {
    // Remove script and style blocks first
    let no_scripts = replace_all(&SCRIPT_RE, html, "");
    let no_scripts = replace_all(&STYLE_RE, &no_scripts, "");
    // Strip remaining tags
    let no_tags = replace_all(&HTML_TAG_RE, &no_scripts, "");
    // Decode common HTML entities
    let decoded = no_tags
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    post_process(&decoded)
}

fn replace_all(
    regex: &LazyLock<Result<Regex, regex::Error>>,
    text: &str,
    replacement: &str,
) -> String {
    match &**regex {
        Ok(regex) => regex.replace_all(text, replacement).into_owned(),
        Err(_) => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn converts_simple_html() {
        let html = "<p>Hello world</p>";
        let result = html_to_markdown(html).await;
        assert!(result.contains("Hello world"));
    }

    #[tokio::test]
    async fn converts_headings() {
        let html = "<h1>Title</h1><p>Content</p>";
        let result = html_to_markdown(html).await;
        assert!(result.contains("Title"));
        assert!(result.contains("Content"));
    }

    #[tokio::test]
    async fn converts_links() {
        let html = r#"<a href="https://example.com">Click here</a>"#;
        let result = html_to_markdown(html).await;
        assert!(result.contains("Click here"));
    }

    #[tokio::test]
    async fn empty_input() {
        assert_eq!(html_to_markdown("").await, "");
        assert_eq!(html_to_markdown("   ").await, "");
    }

    #[test]
    fn fallback_strips_tags() {
        let result = fallback_strip_tags("<b>bold</b> &amp; <i>italic</i>");
        assert!(result.contains("bold"));
        assert!(result.contains("&"));
        assert!(result.contains("italic"));
        assert!(!result.contains("<b>"));
    }

    #[test]
    fn fallback_removes_scripts() {
        let html = "<p>Keep</p><script>alert('x')</script><p>Also keep</p>";
        let result = fallback_strip_tags(html);
        assert!(result.contains("Keep"));
        assert!(result.contains("Also keep"));
        assert!(!result.contains("alert"));
    }
}
