use serde_json::Value;

/// Extract JSON schema (types only, no values).
///
/// Useful for summarizing API responses — shows the shape of the data
/// without the actual values, dramatically reducing token count.
pub fn extract_schema(json_str: &str, max_depth: usize) -> anyhow::Result<String> {
    let value: Value = serde_json::from_str(json_str)?;
    Ok(schema_value(&value, 0, max_depth))
}

/// Compact JSON representation with values preserved but long strings truncated
/// and arrays summarized.
pub fn compact_json(json_str: &str, max_depth: usize) -> anyhow::Result<String> {
    let value: Value = serde_json::from_str(json_str)?;
    Ok(compact_value(&value, 0, max_depth))
}

fn schema_value(value: &Value, depth: usize, max_depth: usize) -> String {
    let indent = "  ".repeat(depth);

    if depth > max_depth {
        return format!("{indent}...");
    }

    match value {
        Value::Null => format!("{indent}null"),
        Value::Bool(_) => format!("{indent}bool"),
        Value::Number(n) => {
            if n.is_i64() {
                format!("{indent}int")
            } else {
                format!("{indent}float")
            }
        }
        Value::String(s) => {
            if s.len() > 50 {
                format!("{indent}string[{}]", s.len())
            } else if s.starts_with("http") {
                format!("{indent}url")
            } else if s.contains('-') && s.len() == 10 {
                format!("{indent}date?")
            } else {
                format!("{indent}string")
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                format!("{indent}[]")
            } else {
                let first_schema = schema_value(&arr[0], depth + 1, max_depth);
                let trimmed = first_schema.trim();
                if arr.len() == 1 {
                    format!("{indent}[\n{first_schema}\n{indent}]")
                } else {
                    format!("{indent}[{trimmed}] ({})", arr.len())
                }
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                format!("{indent}{{}}")
            } else {
                let mut lines = vec![format!("{indent}{{")];
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();

                for (i, key) in keys.iter().enumerate() {
                    let val = &map[*key];
                    let val_schema = schema_value(val, depth + 1, max_depth);
                    let val_trimmed = val_schema.trim();

                    let is_simple = matches!(
                        val,
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
                    );

                    if is_simple {
                        if i < keys.len() - 1 {
                            lines.push(format!("{indent}  {key}: {val_trimmed},"));
                        } else {
                            lines.push(format!("{indent}  {key}: {val_trimmed}"));
                        }
                    } else {
                        lines.push(format!("{indent}  {key}:"));
                        lines.push(val_schema);
                    }

                    if i >= 15 {
                        lines.push(format!("{indent}  ... +{} more keys", keys.len() - i - 1));
                        break;
                    }
                }
                lines.push(format!("{indent}}}"));
                lines.join("\n")
            }
        }
    }
}

fn compact_value(value: &Value, depth: usize, max_depth: usize) -> String {
    let indent = "  ".repeat(depth);

    if depth > max_depth {
        return format!("{indent}...");
    }

    match value {
        Value::Null => format!("{indent}null"),
        Value::Bool(b) => format!("{indent}{b}"),
        Value::Number(n) => format!("{indent}{n}"),
        Value::String(s) => {
            if s.chars().count() > 80 {
                let truncated: String = s.chars().take(77).collect();
                format!("{indent}\"{truncated}...\"")
            } else {
                format!("{indent}\"{s}\"")
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                format!("{indent}[]")
            } else if arr.len() > 5 {
                let first = compact_value(&arr[0], depth + 1, max_depth);
                format!("{indent}[{}, ... +{} more]", first.trim(), arr.len() - 1)
            } else {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| compact_value(v, depth + 1, max_depth))
                    .collect();
                let all_simple = arr.iter().all(|v| {
                    matches!(
                        v,
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
                    )
                });
                if all_simple {
                    let inline: Vec<&str> = items.iter().map(|s| s.trim()).collect();
                    format!("{indent}[{}]", inline.join(", "))
                } else {
                    let mut lines = vec![format!("{indent}[")];
                    for item in &items {
                        lines.push(format!("{item},"));
                    }
                    lines.push(format!("{indent}]"));
                    lines.join("\n")
                }
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                format!("{indent}{{}}")
            } else {
                let mut lines = vec![format!("{indent}{{")];
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();

                for (i, key) in keys.iter().enumerate() {
                    let val = &map[*key];
                    let is_simple = matches!(
                        val,
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
                    );

                    if is_simple {
                        let val_str = compact_value(val, 0, max_depth);
                        lines.push(format!("{indent}  {key}: {}", val_str.trim()));
                    } else {
                        lines.push(format!("{indent}  {key}:"));
                        lines.push(compact_value(val, depth + 1, max_depth));
                    }

                    if i >= 20 {
                        lines.push(format!("{indent}  ... +{} more keys", keys.len() - i - 1));
                        break;
                    }
                }
                lines.push(format!("{indent}}}"));
                lines.join("\n")
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn schema_simple_object() {
        let schema = extract_schema(r#"{"name": "test", "count": 42}"#, 5).unwrap();
        assert!(schema.contains("name"));
        assert!(schema.contains("string"));
        assert!(schema.contains("int"));
    }

    #[test]
    fn schema_array() {
        let schema = extract_schema(r#"{"items": [1, 2, 3]}"#, 5).unwrap();
        assert!(schema.contains("items"));
        assert!(schema.contains("(3)"));
    }

    #[test]
    fn schema_url_detection() {
        let schema = extract_schema(r#"{"link": "https://example.com"}"#, 5).unwrap();
        assert!(schema.contains("url"));
    }

    #[test]
    fn compact_simple() {
        let result = compact_json(r#"{"name": "test", "count": 42}"#, 5).unwrap();
        assert!(result.contains("\"test\""));
        assert!(result.contains("42"));
    }

    #[test]
    fn compact_truncates_long_strings() {
        let long_str = "a".repeat(100);
        let json = format!(r#"{{"data": "{long_str}"}}"#);
        let result = compact_json(&json, 5).unwrap();
        assert!(result.contains("...\""));
    }

    #[test]
    fn compact_summarizes_large_arrays() {
        let json = r#"{"ids": [1, 2, 3, 4, 5, 6, 7, 8]}"#;
        let result = compact_json(json, 5).unwrap();
        assert!(result.contains("+7 more"));
    }

    #[test]
    fn invalid_json_returns_error() {
        assert!(extract_schema("not json", 5).is_err());
        assert!(compact_json("not json", 5).is_err());
    }

    #[test]
    fn schema_nested_object() {
        let json = r#"{"user": {"name": "Alice", "age": 30}}"#;
        let schema = extract_schema(json, 5).unwrap();
        assert!(schema.contains("user"));
        assert!(schema.contains("name"));
        assert!(schema.contains("age"));
        assert!(schema.contains("int"));
    }

    #[test]
    fn schema_depth_limit() {
        let json = r#"{"a": {"b": {"c": {"d": "deep"}}}}"#;
        let schema = extract_schema(json, 1).unwrap();
        assert!(schema.contains("..."));
    }

    #[test]
    fn schema_empty_object() {
        let schema = extract_schema("{}", 5).unwrap();
        assert!(schema.contains("{}"));
    }

    #[test]
    fn schema_empty_array() {
        let schema = extract_schema(r#"{"items": []}"#, 5).unwrap();
        assert!(schema.contains("[]"));
    }

    #[test]
    fn schema_null_bool() {
        let schema = extract_schema(r#"{"a": null, "b": true}"#, 5).unwrap();
        assert!(schema.contains("null"));
        assert!(schema.contains("bool"));
    }

    #[test]
    fn schema_float() {
        let schema = extract_schema(r#"{"pi": 3.14}"#, 5).unwrap();
        assert!(schema.contains("float"));
    }

    #[test]
    fn compact_preserves_small_arrays() {
        let json = r#"{"tags": ["a", "b", "c"]}"#;
        let result = compact_json(json, 5).unwrap();
        // Small arrays (<=5) should show all items inline
        assert!(result.contains("\"a\""));
        assert!(result.contains("\"b\""));
        assert!(result.contains("\"c\""));
        assert!(!result.contains("more"));
    }

    #[test]
    fn compact_empty_structures() {
        let result = compact_json(r#"{"obj": {}, "arr": []}"#, 5).unwrap();
        assert!(result.contains("{}"));
        assert!(result.contains("[]"));
    }

    #[test]
    fn schema_long_string_shows_length() {
        let long = "x".repeat(60);
        let json = format!(r#"{{"data": "{long}"}}"#);
        let schema = extract_schema(&json, 5).unwrap();
        assert!(schema.contains("string[60]"));
    }
}
