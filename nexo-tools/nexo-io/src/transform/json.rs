use serde_json::Value;

/// Compact JSON representation with values preserved but long strings truncated
/// and arrays summarized.
pub fn compact_json(json_str: &str, max_depth: usize) -> anyhow::Result<String> {
    let value: Value = serde_json::from_str(json_str)?;
    Ok(compact_value(&value, 0, max_depth))
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
}
