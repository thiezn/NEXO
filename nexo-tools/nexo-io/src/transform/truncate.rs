/// Truncate a string to `max_len` characters, appending `...` if needed.
pub fn truncate_str(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }
    if max_len < 3 {
        return "...".to_string();
    }
    format!("{}...", s.chars().take(max_len - 3).collect::<String>())
}

/// Truncate output by lines, keeping first and last portions.
///
/// When `output` exceeds `max_lines`, keeps the first half and last half
/// with a `... (N lines omitted)` separator in the middle.
pub fn truncate_lines(output: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= max_lines {
        return output.to_string();
    }

    let head = max_lines / 2;
    let tail = max_lines - head;
    let omitted = lines.len() - head - tail;

    let mut result = String::with_capacity(output.len() / 2);
    for line in &lines[..head] {
        result.push_str(line);
        result.push('\n');
    }
    result.push_str(&format!("\n... ({omitted} lines omitted) ...\n\n"));
    let tail_start = lines.len() - tail;
    for line in &lines[tail_start..] {
        result.push_str(line);
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_over() {
        assert_eq!(truncate_str("hello world", 8), "hello...");
    }

    #[test]
    fn truncate_str_tiny_max() {
        assert_eq!(truncate_str("hello", 2), "...");
    }

    #[test]
    fn truncate_str_unicode() {
        // 5 chars: é, à, ü, ö, ñ
        assert_eq!(truncate_str("éàüöñ", 3), "...");
    }

    #[test]
    fn truncate_lines_within_limit() {
        let input = "a\nb\nc\n";
        assert_eq!(truncate_lines(input, 5), input);
    }

    #[test]
    fn truncate_lines_over_limit() {
        let lines: Vec<String> = (0..20).map(|i| format!("line {i}")).collect();
        let input = lines.join("\n");
        let result = truncate_lines(&input, 6);
        assert!(result.contains("line 0"));
        assert!(result.contains("line 19"));
        assert!(result.contains("lines omitted"));
        // Should contain head (3) + tail (3) + separator
        assert!(!result.contains("line 5"));
    }

    #[test]
    fn truncate_lines_empty() {
        assert_eq!(truncate_lines("", 10), "");
    }
}
