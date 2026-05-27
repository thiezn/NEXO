use serde::{Deserialize, Serialize};

#[cfg(feature = "tool")]
use async_trait::async_trait;

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

/// The side-effect level declared for a tool.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ToolSideEffectLevel {
    /// The tool is read-only and may be parallelized by the orchestrator.
    ReadOnly,
    /// The tool has side effects and must be treated conservatively.
    #[default]
    SideEffecting,
}

/// Scheduling constraints that guide gateway tool orchestration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct ToolExecutionConstraints {
    /// The declared side-effect level for this tool.
    #[serde(default)]
    pub side_effect_level: ToolSideEffectLevel,
    /// Whether the tool may run in parallel with other side-effecting tools.
    #[serde(default)]
    pub parallel_safe: bool,
}

impl ToolExecutionConstraints {
    /// Return whether this value matches the conservative default policy.
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

/// Description of a tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_version: Option<String>,
    #[serde(default, skip_serializing_if = "ToolExecutionConstraints::is_default")]
    pub execution: ToolExecutionConstraints,
}

/// Core tool trait — implement for any capability
#[cfg(feature = "tool")]
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used in LLM function calling)
    fn name(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// JSON schema for parameters
    fn parameters_schema(&self) -> serde_json::Value;

    /// Optional contract version used to detect incompatible tool definitions.
    fn contract_version(&self) -> Option<&str> {
        None
    }

    /// The declared side-effect level for this tool.
    fn side_effect_level(&self) -> ToolSideEffectLevel {
        ToolSideEffectLevel::SideEffecting
    }

    /// Whether this tool is safe to parallelize with other side-effecting tools.
    fn is_parallel_safe(&self) -> bool {
        false
    }

    /// Execute the tool with given arguments
    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;

    /// Get the full spec for LLM registration
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
            contract_version: self.contract_version().map(ToOwned::to_owned),
            execution: ToolExecutionConstraints {
                side_effect_level: self.side_effect_level(),
                parallel_safe: self.is_parallel_safe(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_execution_constraints_default_is_conservative() {
        let constraints = ToolExecutionConstraints::default();

        assert_eq!(
            constraints.side_effect_level,
            ToolSideEffectLevel::SideEffecting
        );
        assert!(!constraints.parallel_safe);
        assert!(constraints.is_default());
    }

    #[test]
    fn tool_spec_omits_default_execution_metadata() {
        let spec = ToolSpec {
            name: "dummy_tool".into(),
            description: "A deterministic test tool".into(),
            parameters: serde_json::json!({"type": "object"}),
            contract_version: None,
            execution: ToolExecutionConstraints::default(),
        };

        let json = serde_json::to_value(&spec).unwrap();
        assert!(json.get("contractVersion").is_none());
        assert!(json.get("execution").is_none());
    }

    #[cfg(feature = "tool")]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tool_trait_tests {
        use super::*;

        struct DummyTool;

        #[async_trait]
        impl Tool for DummyTool {
            fn name(&self) -> &str {
                "dummy_tool"
            }

            fn description(&self) -> &str {
                "A deterministic test tool"
            }

            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                })
            }

            fn contract_version(&self) -> Option<&str> {
                Some("2026-05-22")
            }

            fn side_effect_level(&self) -> ToolSideEffectLevel {
                ToolSideEffectLevel::ReadOnly
            }

            fn is_parallel_safe(&self) -> bool {
                true
            }

            async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    success: true,
                    output: args
                        .get("value")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    error: None,
                })
            }
        }

        #[test]
        fn spec_uses_tool_metadata_and_schema() {
            let tool = DummyTool;
            let spec = tool.spec();

            assert_eq!(spec.name, "dummy_tool");
            assert_eq!(spec.description, "A deterministic test tool");
            assert_eq!(spec.parameters["type"], "object");
            assert_eq!(spec.parameters["properties"]["value"]["type"], "string");
            assert_eq!(spec.contract_version.as_deref(), Some("2026-05-22"));
            assert_eq!(
                spec.execution.side_effect_level,
                ToolSideEffectLevel::ReadOnly
            );
            assert!(spec.execution.parallel_safe);
        }

        #[tokio::test]
        async fn execute_returns_expected_output() {
            let tool = DummyTool;
            let result = tool
                .execute(serde_json::json!({ "value": "hello-tool" }))
                .await
                .unwrap();

            assert!(result.success);
            assert_eq!(result.output, "hello-tool");
            assert!(result.error.is_none());
        }

        #[test]
        fn tool_result_serialization_roundtrip() {
            let result = ToolResult {
                success: false,
                output: String::new(),
                error: Some("boom".into()),
            };

            let json = serde_json::to_string(&result).unwrap();
            let parsed: ToolResult = serde_json::from_str(&json).unwrap();

            assert!(!parsed.success);
            assert_eq!(parsed.error.as_deref(), Some("boom"));
        }
    }
}
