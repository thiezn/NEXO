use async_trait::async_trait;
use nexo_core::{
    Error, Result, Tool, ToolCall, ToolExecutionConstraints, ToolResult, ToolResultContent,
    ToolResultStatus,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct EditArgs {
    #[schemars(description = "File path to edit or create")]
    pub path: String,

    #[schemars(description = "Text to find and replace (omit for file creation)")]
    pub old_string: Option<String>,

    #[schemars(description = "Replacement text (used with old_string)")]
    pub new_string: Option<String>,

    #[schemars(description = "Full file content (for creating new files)")]
    pub content: Option<String>,
}

/// Executes `io.edit` tool against local filesystem
#[derive(Default)]
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    type Args = EditArgs;

    fn name(&self) -> &str {
        "io.edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing text or create a new file with specified content"
    }

    fn execution_constraints(&self) -> ToolExecutionConstraints {
        ToolExecutionConstraints::default_side_effecting()
    }

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        // Initialize arguments
        let args = self.parse_args(&call)?;

        let path = args.path;
        let old_string = args.old_string;
        let new_string = args.new_string;
        let content = args.content;

        // ToolResult helper
        let make_result = |status, content| ToolResult {
            tool_call_id: call.id,
            tool_name: call.name,
            status,
            content: ToolResultContent::Text(content),
        };

        // Perform the file edit or creation in a blocking task to avoid blocking the async runtime.
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
            Ok(output) => make_result(ToolResultStatus::Success, output),
            Err(message) => make_result(ToolResultStatus::Failure, message),
        })
    }
}
