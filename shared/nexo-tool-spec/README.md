# Example of implementing a Tool

Got the code from ZeroClaw. Lets pick and choose various patterns and ideas from them:

The spec https://github.com/zeroclaw-labs/zeroclaw/blob/41dd23175fe991adf1ee1a5693eff69e09ef2a3a/src/tools/traits.rs#L14
An implementation https://github.com/zeroclaw-labs/zeroclaw/blob/41dd23175fe991adf1ee1a5693eff69e09ef2a3a/src/tools/notion_tool.rs#L339


```rust
#[async_trait]
impl Tool for NotionTool {
    fn name(&self) -> &str {
        "notion"
    }

    fn description(&self) -> &str {
        "Interact with Notion: query databases, read/create/update pages, and search the workspace."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query_database", "read_page", "create_page", "update_page", "search"],
                    "description": "The Notion API action to perform"
                },
                "database_id": {
                    "type": "string",
                    "description": "Database ID (required for query_database, optional for create_page)"
                },
                "page_id": {
                    "type": "string",
                    "description": "Page ID (required for read_page and update_page)"
                },
                "filter": {
                    "type": "object",
                    "description": "Notion filter object for query_database"
                },
                "properties": {
                    "type": "object",
                    "description": "Properties object for create_page and update_page"
                },
                "query": {
                    "type": "string",
                    "description": "Search query string for the search action"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let action = match args.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: action".into()),
                });
            }
        };

        // Enforce granular security: Read for queries, Act for mutations
        let operation = match action {
            "query_database" | "read_page" | "search" => ToolOperation::Read,
            "create_page" | "update_page" => ToolOperation::Act,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown action: {action}. Valid actions: query_database, read_page, create_page, update_page, search"
                    )),
                });
            }
        };

        if let Err(error) = self.security.enforce_tool_operation(operation, "notion") {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let result = match action {
            "query_database" => {
                let database_id = match args.get("database_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("query_database requires database_id parameter".into()),
                        });
                    }
                };
                let filter = args.get("filter");
                self.query_database(database_id, filter).await
            }
            "read_page" => {
                let page_id = match args.get("page_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("read_page requires page_id parameter".into()),
                        });
                    }
                };
                self.read_page(page_id).await
            }
            "create_page" => {
                let properties = match args.get("properties") {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("create_page requires properties parameter".into()),
                        });
                    }
                };
                let database_id = args.get("database_id").and_then(|v| v.as_str());
                self.create_page(properties, database_id).await
            }
            "update_page" => {
                let page_id = match args.get("page_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("update_page requires page_id parameter".into()),
                        });
                    }
                };
                let properties = match args.get("properties") {
                    Some(p) => p,
                    None => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some("update_page requires properties parameter".into()),
                        });
                    }
                };
                self.update_page(page_id, properties).await
            }
            "search" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.search(query).await
            }
            _ => unreachable!(), // Already handled above
        };

        match result {
            Ok(value) => Ok(ToolResult {
                success: true,
                output: serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }
}
```
