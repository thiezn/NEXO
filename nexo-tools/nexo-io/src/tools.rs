use std::sync::Arc;

use async_trait::async_trait;
use nexo_spec::tool::{Tool, ToolResult};

use crate::transform;

/// Return all IO tools for gateway-native registration.
///
/// Unlike nexo-notes, these tools operate on the local filesystem and network
/// directly — no storage dependency needed.
pub fn all_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(IoRead),
        Arc::new(IoEdit),
        Arc::new(IoBash),
        Arc::new(IoWebFetch {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("nexo-io/0.1")
                .build()
                .expect("valid HTTP client config"),
        }),
    ]
}

// ── io.read ───────────────────────────────────────────────────────────────────

struct IoRead;

#[async_trait]
impl Tool for IoRead {
    fn name(&self) -> &str {
        "io.read"
    }

    fn description(&self) -> &str {
        "Read a file from the local filesystem. Returns file content with optional \
         offset/limit for partial reads. Applies language-aware filtering to strip \
         comments and normalize formatting."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?
            .to_string();
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        tokio::task::spawn_blocking(move || {
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to read {path}: {e}")),
                    });
                }
            };

            // Detect language and apply minimal filter
            let lang = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .map(transform::code_filter::Language::from_extension)
                .unwrap_or(transform::code_filter::Language::Unknown);

            let mut filtered = transform::code_filter::minimal_filter(&content, &lang);

            // Safety: if filter emptied a non-empty file, fall back to raw
            if filtered.trim().is_empty() && !content.trim().is_empty() {
                filtered = content;
            }

            // Apply offset/limit
            let lines: Vec<&str> = filtered.lines().collect();
            let start = offset.min(lines.len());
            let end = match limit {
                Some(lim) => (start + lim).min(lines.len()),
                None => lines.len(),
            };
            let sliced = lines[start..end].join("\n");

            // Strip any ANSI codes
            let output = transform::ansi::strip_ansi(&sliced);

            Ok(ToolResult {
                success: true,
                output,
                error: None,
            })
        })
        .await?
    }
}

// ── io.edit ───────────────────────────────────────────────────────────────────

struct IoEdit;

#[async_trait]
impl Tool for IoEdit {
    fn name(&self) -> &str {
        "io.edit"
    }

    fn description(&self) -> &str {
        "Edit an existing file or create a new file. For editing: provide path, \
         old_string (text to find), and new_string (replacement). For creating: \
         provide path and content."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to edit or create"
                },
                "old_string": {
                    "type": "string",
                    "description": "Text to find and replace (omit for file creation)"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text (used with old_string)"
                },
                "content": {
                    "type": "string",
                    "description": "Full file content (for creating new files)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?
            .to_string();
        let old_string = args
            .get("old_string")
            .and_then(|v| v.as_str())
            .map(String::from);
        let new_string = args
            .get("new_string")
            .and_then(|v| v.as_str())
            .map(String::from);
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from);

        tokio::task::spawn_blocking(move || {
            let p = std::path::Path::new(&path);

            // Create mode: content present, no old_string
            if let Some(content) = content {
                if old_string.is_some() {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "Cannot use both 'content' (create) and 'old_string' (edit) simultaneously"
                                .into(),
                        ),
                    });
                }
                if let Some(parent) = p.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some(format!("Failed to create directory: {e}")),
                        });
                    }
                }
                if let Err(e) = std::fs::write(p, &content) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to write {path}: {e}")),
                    });
                }
                return Ok(ToolResult {
                    success: true,
                    output: format!("Created {path} ({} bytes)", content.len()),
                    error: None,
                });
            }

            // Edit mode: old_string present
            if let Some(old_string) = old_string {
                let new_string = new_string.unwrap_or_default();

                let file_content = match std::fs::read_to_string(p) {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some(format!("Failed to read {path}: {e}")),
                        });
                    }
                };

                let pos = match file_content.find(&old_string) {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("old_string not found in file".into()),
                        });
                    }
                };

                // Check for a second occurrence
                if file_content[pos + old_string.len()..].contains(&old_string) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "old_string found multiple times; provide more surrounding context for a unique match"
                                .into(),
                        ),
                    });
                }

                let mut updated = String::with_capacity(
                    file_content.len() - old_string.len() + new_string.len(),
                );
                updated.push_str(&file_content[..pos]);
                updated.push_str(&new_string);
                updated.push_str(&file_content[pos + old_string.len()..]);
                if let Err(e) = std::fs::write(p, &updated) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to write {path}: {e}")),
                    });
                }

                return Ok(ToolResult {
                    success: true,
                    output: format!("Edited {path}: replaced 1 occurrence"),
                    error: None,
                });
            }

            // Neither mode
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "Provide either 'content' (to create a file) or 'old_string' (to edit a file)"
                        .into(),
                ),
            })
        })
        .await?
    }
}

// ── io.bash ───────────────────────────────────────────────────────────────────

struct IoBash;

#[async_trait]
impl Tool for IoBash {
    fn name(&self) -> &str {
        "io.bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command on the gateway host. Returns stdout, stderr, and exit code. \
         Output is cleaned (ANSI stripped) and truncated if very large."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000, max: 120000)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: command"))?
            .to_string();
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(30_000)
            .clamp(1_000, 120_000);

        let timeout = std::time::Duration::from_millis(timeout_ms);

        let result = tokio::time::timeout(timeout, async {
            let child = tokio::process::Command::new("bash")
                .arg("-c")
                .arg(&command)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()?;
            child.wait_with_output().await
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

                let stdout = transform::ansi::strip_ansi(&stdout);
                let stderr = transform::ansi::strip_ansi(&stderr);

                let stdout = transform::truncate::truncate_lines(&stdout, 500);
                let stderr = transform::truncate::truncate_lines(&stderr, 500);

                let mut out = format!("exit_code: {exit_code}\n");
                if !stdout.trim().is_empty() {
                    out.push_str("stdout:\n");
                    out.push_str(&stdout);
                }
                if !stderr.trim().is_empty() {
                    if !stdout.trim().is_empty() {
                        out.push('\n');
                    }
                    out.push_str("stderr:\n");
                    out.push_str(&stderr);
                }

                Ok(ToolResult {
                    success: exit_code == 0,
                    output: out,
                    error: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to execute command: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Command timed out after {timeout_ms}ms")),
            }),
        }
    }
}

// ── io.web_fetch ──────────────────────────────────────────────────────────────

struct IoWebFetch {
    client: reqwest::Client,
}

#[async_trait]
impl Tool for IoWebFetch {
    fn name(&self) -> &str {
        "io.web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a web page or API endpoint. HTML responses are converted to readable \
         markdown. JSON responses are compacted. Plain text is returned as-is."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: url"))?
            .to_string();

        let response = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Request failed: {e}")),
                });
            }
        };

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("HTTP {status}")),
            });
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let body = match response.text().await {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read response body: {e}")),
                });
            }
        };

        let output = if content_type.contains("text/html") {
            let md = transform::html::html_to_markdown(&body).await;
            transform::truncate::truncate_lines(&md, 500)
        } else if content_type.contains("application/json") {
            match transform::json::compact_json(&body, 5) {
                Ok(compact) => compact,
                Err(_) => transform::truncate::truncate_lines(&body, 500),
            }
        } else {
            transform::truncate::truncate_lines(&body, 500)
        };

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_nonexistent_file() {
        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read
            .execute(serde_json::json!({"path": "/nonexistent/file.txt"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Failed to read"));
    }

    #[tokio::test]
    async fn read_real_file() {
        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read
            .execute(serde_json::json!({"path": "/etc/hosts"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(!result.output.is_empty());
    }

    #[tokio::test]
    async fn read_with_offset_limit() {
        // Create a temp file
        let dir = std::env::temp_dir().join("nexo-io-test-read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        std::fs::write(&path, "line0\nline1\nline2\nline3\nline4\n").unwrap();

        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "offset": 1,
                "limit": 2
            }))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("line1"));
        assert!(result.output.contains("line2"));
        assert!(!result.output.contains("line0"));
        assert!(!result.output.contains("line3"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_create_file() {
        let dir = std::env::temp_dir().join("nexo-io-test-create");
        let path = dir.join("new.txt");

        // Clean up from previous runs
        std::fs::remove_dir_all(&dir).ok();

        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "content": "hello world"
            }))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Created"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello world");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_replace_string() {
        let dir = std::env::temp_dir().join("nexo-io-test-edit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        std::fs::write(&path, "foo bar baz").unwrap();

        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "bar",
                "new_string": "qux"
            }))
            .await
            .unwrap();
        assert!(result.success);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "foo qux baz");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_ambiguous_match() {
        let dir = std::env::temp_dir().join("nexo-io-test-ambiguous");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dup.txt");
        std::fs::write(&path, "aa bb aa").unwrap();

        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "aa",
                "new_string": "cc"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("multiple times"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_not_found() {
        let dir = std::env::temp_dir().join("nexo-io-test-notfound");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nf.txt");
        std::fs::write(&path, "hello").unwrap();

        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "old_string": "nonexistent",
                "new_string": "x"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn bash_echo() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("exit_code: 0"));
    }

    #[tokio::test]
    async fn bash_failing_command() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash
            .execute(serde_json::json!({"command": "exit 42"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.output.contains("exit_code: 42"));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash
            .execute(serde_json::json!({
                "command": "sleep 60",
                "timeout_ms": 1000
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn all_tools_registered() {
        let tools = all_tools();
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"io.read"));
        assert!(names.contains(&"io.edit"));
        assert!(names.contains(&"io.bash"));
        assert!(names.contains(&"io.web_fetch"));
    }

    #[tokio::test]
    async fn all_tools_have_specs() {
        for tool in all_tools() {
            let spec = tool.spec();
            assert!(!spec.name.is_empty());
            assert!(!spec.description.is_empty());
            assert!(spec.parameters.is_object());
        }
    }

    #[tokio::test]
    async fn read_strips_comments_from_rust() {
        let dir = std::env::temp_dir().join("nexo-io-test-filter");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("code.rs");
        std::fs::write(&path, "// comment\nfn main() {}\n").unwrap();

        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read
            .execute(serde_json::json!({"path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(!result.output.contains("// comment"));
        assert!(result.output.contains("fn main()"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_preserves_json_content() {
        let dir = std::env::temp_dir().join("nexo-io-test-json-read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.json");
        std::fs::write(&path, r#"{"packages": ["pkg/*"]}"#).unwrap();

        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read
            .execute(serde_json::json!({"path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("pkg/*"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_no_params_returns_error() {
        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({"path": "/tmp/whatever.txt"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Provide either"));
    }

    #[tokio::test]
    async fn edit_content_and_old_string_conflict() {
        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": "/tmp/conflict.txt",
                "content": "new file",
                "old_string": "foo"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Cannot use both"));
    }

    #[tokio::test]
    async fn edit_creates_nested_dirs() {
        let dir = std::env::temp_dir().join("nexo-io-test-nested/a/b/c");
        let path = dir.join("deep.txt");
        std::fs::remove_dir_all(std::env::temp_dir().join("nexo-io-test-nested")).ok();

        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "content": "deep content"
            }))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "deep content");

        std::fs::remove_dir_all(std::env::temp_dir().join("nexo-io-test-nested")).ok();
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash
            .execute(serde_json::json!({"command": "echo err >&2"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("stderr:"));
        assert!(result.output.contains("err"));
    }

    #[tokio::test]
    async fn bash_strips_ansi() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash
            .execute(serde_json::json!({"command": "printf '\\x1b[31mred\\x1b[0m'"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("red"));
        assert!(!result.output.contains("\x1b["));
    }

    #[tokio::test]
    async fn bash_missing_command_param() {
        let tools = all_tools();
        let bash = tools.iter().find(|t| t.name() == "io.bash").unwrap();
        let result = bash.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn web_fetch_missing_url() {
        let tools = all_tools();
        let fetch = tools.iter().find(|t| t.name() == "io.web_fetch").unwrap();
        let result = fetch.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_missing_path() {
        let tools = all_tools();
        let read = tools.iter().find(|t| t.name() == "io.read").unwrap();
        let result = read.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn edit_missing_path() {
        let tools = all_tools();
        let edit = tools.iter().find(|t| t.name() == "io.edit").unwrap();
        let result = edit.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }
}
