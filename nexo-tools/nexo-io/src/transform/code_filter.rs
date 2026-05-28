/// Programming language classification for comment-aware filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Rust source files.
    Rust,
    /// Python source files.
    Python,
    /// JavaScript source files.
    JavaScript,
    /// TypeScript source files.
    TypeScript,
    /// Go source files.
    Go,
    /// C source or header files.
    C,
    /// C++ source or header files.
    Cpp,
    /// Java source files.
    Java,
    /// Ruby source files.
    Ruby,
    /// Shell scripts.
    Shell,
    /// Swift source files.
    Swift,
    /// Data formats (JSON, YAML, TOML, XML, CSV, etc.) — no comment stripping.
    Data,
    /// Unknown or unsupported file types.
    Unknown,
}

impl Language {
    /// Detect language from a file extension (case-insensitive).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" | "pyw" => Self::Python,
            "js" | "mjs" | "cjs" | "jsx" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "go" => Self::Go,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hh" => Self::Cpp,
            "java" => Self::Java,
            "rb" => Self::Ruby,
            "sh" | "bash" | "zsh" => Self::Shell,
            "swift" => Self::Swift,
            "json" | "jsonc" | "json5" | "yaml" | "yml" | "toml" | "xml" | "csv" | "tsv"
            | "graphql" | "gql" | "sql" | "md" | "markdown" | "txt" | "env" | "lock" => Self::Data,
            _ => Self::Unknown,
        }
    }

    /// Return the comment syntax patterns for this language.
    pub fn comment_patterns(&self) -> CommentPatterns {
        match self {
            Self::Rust => CommentPatterns {
                line: Some("//"),
                block_start: Some("/*"),
                block_end: Some("*/"),
                doc_line: Some("///"),
                doc_block_start: Some("/**"),
            },
            Self::Python => CommentPatterns {
                line: Some("#"),
                block_start: Some("\"\"\""),
                block_end: Some("\"\"\""),
                doc_line: None,
                doc_block_start: Some("\"\"\""),
            },
            Self::JavaScript
            | Self::TypeScript
            | Self::Go
            | Self::C
            | Self::Cpp
            | Self::Java
            | Self::Swift => CommentPatterns {
                line: Some("//"),
                block_start: Some("/*"),
                block_end: Some("*/"),
                doc_line: None,
                doc_block_start: Some("/**"),
            },
            Self::Ruby => CommentPatterns {
                line: Some("#"),
                block_start: Some("=begin"),
                block_end: Some("=end"),
                doc_line: None,
                doc_block_start: None,
            },
            Self::Shell => CommentPatterns {
                line: Some("#"),
                block_start: None,
                block_end: None,
                doc_line: None,
                doc_block_start: None,
            },
            Self::Data => CommentPatterns {
                line: None,
                block_start: None,
                block_end: None,
                doc_line: None,
                doc_block_start: None,
            },
            Self::Unknown => CommentPatterns {
                line: Some("//"),
                block_start: Some("/*"),
                block_end: Some("*/"),
                doc_line: None,
                doc_block_start: None,
            },
        }
    }
}

/// Comment syntax patterns for a specific language.
#[derive(Debug, Clone)]
pub struct CommentPatterns {
    /// Prefix for single-line comments.
    pub line: Option<&'static str>,
    /// Prefix for block comment start marker.
    pub block_start: Option<&'static str>,
    /// Suffix for block comment end marker.
    pub block_end: Option<&'static str>,
    /// Prefix for single-line documentation comments.
    pub doc_line: Option<&'static str>,
    /// Prefix for block documentation comments.
    pub doc_block_start: Option<&'static str>,
}

/// Apply minimal code filtering: strip non-doc comments, normalize blank lines.
///
/// Data formats (JSON, YAML, TOML, XML, CSV, Markdown) pass through unchanged
/// to avoid corrupting content that uses comment-like syntax (e.g. `/*` in JSON values).
pub fn minimal_filter(content: &str, lang: &Language) -> String {
    // Data formats must never have comments removed
    if *lang == Language::Data {
        return content.to_string();
    }

    let patterns = lang.comment_patterns();
    let mut result = String::with_capacity(content.len());
    let mut in_block_comment = false;
    let mut in_docstring = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Handle block comments
        if let (Some(start), Some(end)) = (patterns.block_start, patterns.block_end) {
            if !in_docstring
                && trimmed.contains(start)
                && !trimmed.starts_with(patterns.doc_block_start.unwrap_or("###"))
            {
                in_block_comment = true;
            }
            if in_block_comment {
                if trimmed.contains(end) {
                    in_block_comment = false;
                }
                continue;
            }
        }

        // Handle Python docstrings (keep them)
        if *lang == Language::Python && trimmed.starts_with("\"\"\"") {
            in_docstring = !in_docstring;
            result.push_str(line);
            result.push('\n');
            continue;
        }

        if in_docstring {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // Skip single-line comments (but keep doc comments)
        if let Some(line_comment) = patterns.line
            && trimmed.starts_with(line_comment)
        {
            // Keep doc comments (e.g. /// in Rust)
            if let Some(doc) = patterns.doc_line
                && trimmed.starts_with(doc)
            {
                result.push_str(line);
                result.push('\n');
            }
            continue;
        }

        // Blank lines pass through (normalized later)
        if trimmed.is_empty() {
            result.push('\n');
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    // Normalize multiple blank lines to max 2
    let result = super::normalize_blank_lines(&result);
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_rust_comments() {
        let code = "// This is a comment\nfn main() {\n    println!(\"Hello\");\n}\n";
        let result = minimal_filter(code, &Language::Rust);
        assert!(!result.contains("// This is a comment"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn keeps_rust_doc_comments() {
        let code = "/// Doc comment\nfn foo() {}\n";
        let result = minimal_filter(code, &Language::Rust);
        assert!(result.contains("/// Doc comment"));
    }

    #[test]
    fn strips_block_comments() {
        let code = "/* block\n   comment */\nfn bar() {}\n";
        let result = minimal_filter(code, &Language::Rust);
        assert!(!result.contains("block"));
        assert!(result.contains("fn bar()"));
    }

    #[test]
    fn data_formats_pass_through() {
        let json = r#"{"workspaces": {"packages": ["packages/*"]}}"#;
        let result = minimal_filter(json, &Language::Data);
        assert_eq!(result, json);
    }

    #[test]
    fn json_slash_star_not_treated_as_comment() {
        // RTK #464: /* in JSON values must not be treated as block comment
        let json = r#"{
  "workspaces": {
    "packages": [
      "packages/*"
    ]
  },
  "scripts": {
    "build": "bun run build"
  }
}"#;
        let result = minimal_filter(json, &Language::Data);
        assert!(result.contains("packages/*"));
        assert!(result.contains("scripts"));
    }

    #[test]
    fn python_docstrings_preserved() {
        let code = "def foo():\n    \"\"\"Docstring.\"\"\"\n    pass\n";
        let result = minimal_filter(code, &Language::Python);
        assert!(result.contains("\"\"\"Docstring.\"\"\""));
    }

    #[test]
    fn python_hash_comments_stripped() {
        let code = "# comment\nx = 1\n";
        let result = minimal_filter(code, &Language::Python);
        assert!(!result.contains("# comment"));
        assert!(result.contains("x = 1"));
    }

    #[test]
    fn language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("swift"), Language::Swift);
        assert_eq!(Language::from_extension("json"), Language::Data);
        assert_eq!(Language::from_extension("yml"), Language::Data);
        assert_eq!(Language::from_extension("xyz"), Language::Unknown);
    }

    #[test]
    fn normalizes_blank_lines() {
        let code = "a\n\n\n\n\nb\n";
        let result = minimal_filter(code, &Language::Rust);
        assert!(!result.contains("\n\n\n"));
    }

    #[test]
    fn swift_comments_stripped() {
        let code = "// Swift comment\nfunc greet() {\n    print(\"Hi\")\n}\n";
        let result = minimal_filter(code, &Language::Swift);
        assert!(!result.contains("// Swift comment"));
        assert!(result.contains("func greet()"));
    }

    #[test]
    fn shell_comments_stripped() {
        let code = "#!/bin/bash\n# comment\necho hello\n";
        let result = minimal_filter(code, &Language::Shell);
        assert!(!result.contains("# comment"));
        assert!(result.contains("echo hello"));
    }

    #[test]
    fn preserves_code_with_inline_slashes() {
        // Ensure URLs and paths inside code aren't stripped
        let code = "let url = \"https://example.com/path\";\n";
        let result = minimal_filter(code, &Language::Rust);
        assert!(result.contains("https://example.com/path"));
    }

    #[test]
    fn empty_input() {
        assert_eq!(minimal_filter("", &Language::Rust), "");
    }

    #[test]
    fn all_languages_have_comment_patterns() {
        // Ensure no panic when calling comment_patterns on every variant
        let langs = [
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::TypeScript,
            Language::Go,
            Language::C,
            Language::Cpp,
            Language::Java,
            Language::Ruby,
            Language::Shell,
            Language::Swift,
            Language::Data,
            Language::Unknown,
        ];
        for lang in &langs {
            let _ = lang.comment_patterns();
        }
    }

    #[test]
    fn case_insensitive_extension() {
        assert_eq!(Language::from_extension("RS"), Language::Rust);
        assert_eq!(Language::from_extension("Py"), Language::Python);
        assert_eq!(Language::from_extension("JSON"), Language::Data);
    }
}
