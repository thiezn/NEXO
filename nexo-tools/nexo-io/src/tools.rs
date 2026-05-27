use std::collections::BTreeMap;

use nexo_core::{
    Error, ToolCall, ToolDefinition, ToolExecutionConstraints, ToolExecutor, ToolParallelism,
    ToolResult, ToolResultContent, ToolResultStatus, ToolSideEffectLevel,
};

use crate::transform;

/// Return all IO tools for gateway-native registration.
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "io.read".to_string(),
            description: "Read a file from the local filesystem. Returns file content with optional offset/limit for partial reads. Applies language-aware filtering to strip comments and normalize formatting.".to_string(),
            parameters: serde_json::json!({
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
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "io.edit".to_string(),
            description: "Edit an existing file or create a new file. For editing: provide path, old_string (text to find), and new_string (replacement). For creating: provide path and content.".to_string(),
            parameters: serde_json::json!({
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
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::SideEffecting,
                parallelism: ToolParallelism::Sequential,
                timeout_ms: None,
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "io.bash".to_string(),
            description: "Execute a bash command on the gateway host. Returns stdout, stderr, and exit code. Output is cleaned (ANSI stripped) and truncated if very large.".to_string(),
            parameters: serde_json::json!({
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
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::SideEffecting,
                parallelism: ToolParallelism::Sequential,
                timeout_ms: Some(120_000),
            },
            metadata: BTreeMap::new(),
        },
        ToolDefinition {
            name: "io.web_fetch".to_string(),
            description: "Fetch a web page or API endpoint. HTML responses are converted to readable markdown. JSON responses are compacted. Plain text is returned as-is.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    }
                },
                "required": ["url"]
            }),
            contract_version: None,
            execution: ToolExecutionConstraints {
                side_effect_level: ToolSideEffectLevel::ReadOnly,
                parallelism: ToolParallelism::ParallelGlobal,
                timeout_ms: Some(30_000),
            },
            metadata: BTreeMap::new(),
        },
    ]
}

/// Executes `io.*` tools against local filesystem, shell, and HTTP endpoints.
pub struct IoToolExecutor {
    client: reqwest::Client,
}

impl IoToolExecutor {
    /// Create a new IO tool executor with a default HTTP client.
    ///
    /// The internal client uses a 30-second timeout and a `nexo-io/0.1`
    /// user-agent string for remote fetch operations.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("nexo-io/0.1")
                .build()
                .expect("valid HTTP client config"),
        }
    }

    async fn execute_read(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let path = call
            .arguments
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: path".to_string(),
            })?
            .to_string();
        let offset = call
            .arguments
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let limit = call
            .arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let outcome = tokio::task::spawn_blocking(move || {
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => return Err(format!("Failed to read {path}: {e}")),
            };

            let lang = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .map(transform::code_filter::Language::from_extension)
                .unwrap_or(transform::code_filter::Language::Unknown);

            let mut filtered = transform::code_filter::minimal_filter(&content, &lang);
            if filtered.trim().is_empty() && !content.trim().is_empty() {
                filtered = content;
            }

            let lines: Vec<&str> = filtered.lines().collect();
            let start = offset.min(lines.len());
            let end = match limit {
                Some(lim) => (start + lim).min(lines.len()),
                None => lines.len(),
            };

            let sliced = lines[start..end].join("\n");
            Ok(transform::ansi::strip_ansi(&sliced))
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("io.read join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(message) => fail_text(call, message),
        })
    }

    async fn execute_edit(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let path = call
            .arguments
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: path".to_string(),
            })?
            .to_string();
        let old_string = call
            .arguments
            .get("old_string")
            .and_then(|v| v.as_str())
            .map(String::from);
        let new_string = call
            .arguments
            .get("new_string")
            .and_then(|v| v.as_str())
            .map(String::from);
        let content = call
            .arguments
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from);

        let outcome = tokio::task::spawn_blocking(move || {
            let p = std::path::Path::new(&path);

            if let Some(content) = content {
                if old_string.is_some() {
                    return Err("Cannot use both 'content' (create) and 'old_string' (edit) simultaneously".to_string());
                }

                if let Some(parent) = p.parent()
                    && let Err(e) = std::fs::create_dir_all(parent)
                {
                    return Err(format!("Failed to create directory: {e}"));
                }

                if let Err(e) = std::fs::write(p, &content) {
                    return Err(format!("Failed to write {path}: {e}"));
                }

                return Ok(format!("Created {path} ({} bytes)", content.len()));
            }

            if let Some(old_string) = old_string {
                let new_string = new_string.unwrap_or_default();
                let file_content =
                    std::fs::read_to_string(p).map_err(|e| format!("Failed to read {path}: {e}"))?;

                let pos = file_content
                    .find(&old_string)
                    .ok_or_else(|| "old_string not found in file".to_string())?;

                if file_content[pos + old_string.len()..].contains(&old_string) {
                    return Err(
                        "old_string found multiple times; provide more surrounding context for a unique match"
                            .to_string(),
                    );
                }

                let mut updated =
                    String::with_capacity(file_content.len() - old_string.len() + new_string.len());
                updated.push_str(&file_content[..pos]);
                updated.push_str(&new_string);
                updated.push_str(&file_content[pos + old_string.len()..]);

                if let Err(e) = std::fs::write(p, &updated) {
                    return Err(format!("Failed to write {path}: {e}"));
                }

                return Ok(format!("Edited {path}: replaced 1 occurrence"));
            }

            Err(
                "Provide either 'content' (to create a file) or 'old_string' (to edit a file)"
                    .to_string(),
            )
        })
        .await
        .map_err(|e| Error::InvalidState {
            message: format!("io.edit join error: {e}"),
        })?;

        Ok(match outcome {
            Ok(output) => ok_text(call, output),
            Err(message) => fail_text(call, message),
        })
    }

    async fn execute_bash(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let command = call
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: command".to_string(),
            })?
            .to_string();

        let timeout_ms = call
            .arguments
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

        Ok(match result {
            Ok(Ok(output)) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = transform::ansi::strip_ansi(&String::from_utf8_lossy(&output.stdout));
                let stderr = transform::ansi::strip_ansi(&String::from_utf8_lossy(&output.stderr));
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

                if exit_code == 0 {
                    ok_text(call, out)
                } else {
                    fail_text(call, out)
                }
            }
            Ok(Err(e)) => fail_text(call, format!("Failed to execute command: {e}")),
            Err(_) => fail_text(call, format!("Command timed out after {timeout_ms}ms")),
        })
    }

    async fn execute_web_fetch(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        let url = call
            .arguments
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidRequest {
                message: "missing required parameter: url".to_string(),
            })?
            .to_string();

        let response = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => return Ok(fail_text(call, format!("Request failed: {e}"))),
        };

        let status = response.status();
        if !status.is_success() {
            return Ok(fail_text(call, format!("HTTP {status}")));
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
                return Ok(fail_text(
                    call,
                    format!("Failed to read response body: {e}"),
                ));
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

        Ok(ok_text(call, output))
    }
}

impl Default for IoToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolExecutor for IoToolExecutor {
    async fn execute(&self, call: ToolCall) -> nexo_core::Result<ToolResult> {
        match call.name.as_str() {
            "io.read" => self.execute_read(call).await,
            "io.edit" => self.execute_edit(call).await,
            "io.bash" => self.execute_bash(call).await,
            "io.web_fetch" => self.execute_web_fetch(call).await,
            _ => Err(Error::UnsupportedFeature {
                feature: format!("unknown tool: {}", call.name),
            }),
        }
    }
}

fn ok_text(call: ToolCall, output: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id,
        tool_name: call.name,
        status: ToolResultStatus::Success,
        content: ToolResultContent::Text(output),
    }
}

fn fail_text(call: ToolCall, message: String) -> ToolResult {
    ToolResult {
        tool_call_id: call.id,
        tool_name: call.name,
        status: ToolResultStatus::Failure,
        content: ToolResultContent::Text(message),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "tc-1".into(),
            index: 0,
            name: name.to_string(),
            arguments: args,
        }
    }

    fn text_output(result: &ToolResult) -> &str {
        match &result.content {
            ToolResultContent::Text(value) => value.as_str(),
            ToolResultContent::Json(_) => panic!("expected text output"),
        }
    }

    #[tokio::test]
    async fn read_nonexistent_file() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.read",
                serde_json::json!({"path": "/nonexistent/file.txt"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("Failed to read"));
    }

    #[tokio::test]
    async fn read_real_file() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call("io.read", serde_json::json!({"path": "/etc/hosts"})))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(!text_output(&result).is_empty());
    }

    #[tokio::test]
    async fn read_with_offset_limit() {
        let dir = std::env::temp_dir().join("nexo-io-test-read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        std::fs::write(&path, "line0\nline1\nline2\nline3\nline4\n").unwrap();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.read",
                serde_json::json!({
                    "path": path.to_str().unwrap(),
                    "offset": 1,
                    "limit": 2
                }),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("line1"));
        assert!(text_output(&result).contains("line2"));
        assert!(!text_output(&result).contains("line0"));
        assert!(!text_output(&result).contains("line3"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_create_file() {
        let dir = std::env::temp_dir().join("nexo-io-test-create");
        let path = dir.join("new.txt");
        std::fs::remove_dir_all(&dir).ok();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": path.to_str().unwrap(), "content": "hello world"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("Created"));

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

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": path.to_str().unwrap(), "old_string": "bar", "new_string": "qux"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);

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

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": path.to_str().unwrap(), "old_string": "aa", "new_string": "cc"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("multiple times"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_not_found() {
        let dir = std::env::temp_dir().join("nexo-io-test-notfound");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nf.txt");
        std::fs::write(&path, "hello").unwrap();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": path.to_str().unwrap(), "old_string": "nonexistent", "new_string": "x"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("not found"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn bash_echo() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.bash",
                serde_json::json!({"command": "echo hello"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("hello"));
        assert!(text_output(&result).contains("exit_code: 0"));
    }

    #[tokio::test]
    async fn bash_failing_command() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call("io.bash", serde_json::json!({"command": "exit 42"})))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("exit_code: 42"));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.bash",
                serde_json::json!({"command": "sleep 60", "timeout_ms": 1000}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("timed out"));
    }

    #[tokio::test]
    async fn all_tools_registered() {
        let tools = all_tools();
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"io.read"));
        assert!(names.contains(&"io.edit"));
        assert!(names.contains(&"io.bash"));
        assert!(names.contains(&"io.web_fetch"));
    }

    #[tokio::test]
    async fn all_tools_have_specs() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
            assert!(tool.parameters.is_object());
        }
    }

    #[tokio::test]
    async fn read_strips_comments_from_rust() {
        let dir = std::env::temp_dir().join("nexo-io-test-filter");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("code.rs");
        std::fs::write(&path, "// comment\nfn main() {}\n").unwrap();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.read",
                serde_json::json!({"path": path.to_str().unwrap()}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(!text_output(&result).contains("// comment"));
        assert!(text_output(&result).contains("fn main()"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_preserves_json_content() {
        let dir = std::env::temp_dir().join("nexo-io-test-json-read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.json");
        std::fs::write(&path, r#"{"packages": ["pkg/*"]}"#).unwrap();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.read",
                serde_json::json!({"path": path.to_str().unwrap()}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("pkg/*"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_no_params_returns_error() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": "/tmp/whatever.txt"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("Provide either"));
    }

    #[tokio::test]
    async fn edit_content_and_old_string_conflict() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": "/tmp/conflict.txt", "content": "new file", "old_string": "foo"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Failure);
        assert!(text_output(&result).contains("Cannot use both"));
    }

    #[tokio::test]
    async fn edit_creates_nested_dirs() {
        let dir = std::env::temp_dir().join("nexo-io-test-nested/a/b/c");
        let path = dir.join("deep.txt");
        std::fs::remove_dir_all(std::env::temp_dir().join("nexo-io-test-nested")).ok();

        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.edit",
                serde_json::json!({"path": path.to_str().unwrap(), "content": "deep content"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "deep content");

        std::fs::remove_dir_all(std::env::temp_dir().join("nexo-io-test-nested")).ok();
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.bash",
                serde_json::json!({"command": "echo err >&2"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("stderr:"));
        assert!(text_output(&result).contains("err"));
    }

    #[tokio::test]
    async fn bash_strips_ansi() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call(
                "io.bash",
                serde_json::json!({"command": "printf '\\x1b[31mred\\x1b[0m'"}),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, ToolResultStatus::Success);
        assert!(text_output(&result).contains("red"));
        assert!(!text_output(&result).contains("\x1b["));
    }

    #[tokio::test]
    async fn bash_missing_command_param() {
        let exec = IoToolExecutor::new();
        let result = exec.execute(call("io.bash", serde_json::json!({}))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn web_fetch_missing_url() {
        let exec = IoToolExecutor::new();
        let result = exec
            .execute(call("io.web_fetch", serde_json::json!({})))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_missing_path() {
        let exec = IoToolExecutor::new();
        let result = exec.execute(call("io.read", serde_json::json!({}))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn edit_missing_path() {
        let exec = IoToolExecutor::new();
        let result = exec.execute(call("io.edit", serde_json::json!({}))).await;
        assert!(result.is_err());
    }
}
