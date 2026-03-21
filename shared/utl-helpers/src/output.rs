use crate::error::{Error, Result};
use clap::ValueEnum;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Json,
    Markdown,
}

/// A single segment in a field path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PathSegment {
    Key(String),
    Index(usize),
    Wildcard,
}

/// A parsed field path such as `results[*].response.status`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FieldPath {
    segments: Vec<PathSegment>,
}

impl FieldPath {
    fn parse(input: &str) -> Self {
        let mut segments = Vec::new();
        for part in input.split('.') {
            if let Some(bracket_pos) = part.find('[') {
                let key = &part[..bracket_pos];
                if !key.is_empty() {
                    segments.push(PathSegment::Key(key.to_string()));
                }
                let bracket_content = &part[bracket_pos + 1..part.len() - 1];
                if bracket_content == "*" {
                    segments.push(PathSegment::Wildcard);
                } else if let Ok(idx) = bracket_content.parse::<usize>() {
                    segments.push(PathSegment::Index(idx));
                }
            } else {
                segments.push(PathSegment::Key(part.to_string()));
            }
        }
        FieldPath { segments }
    }
}

fn extract_path(
    value: &serde_json::Value,
    segments: &[PathSegment],
    prefix: &str,
) -> Vec<(String, serde_json::Value)> {
    if segments.is_empty() {
        return vec![(prefix.to_string(), value.clone())];
    }

    match &segments[0] {
        PathSegment::Key(key) => {
            if let Some(child) = value.get(key.as_str()) {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                extract_path(child, &segments[1..], &new_prefix)
            } else {
                vec![]
            }
        }
        PathSegment::Index(idx) => {
            if let Some(child) = value.get(*idx) {
                let new_prefix = if prefix.is_empty() {
                    format!("[{idx}]")
                } else {
                    format!("{prefix}[{idx}]")
                };
                extract_path(child, &segments[1..], &new_prefix)
            } else {
                vec![]
            }
        }
        PathSegment::Wildcard => {
            if let serde_json::Value::Array(items) = value {
                items
                    .iter()
                    .enumerate()
                    .flat_map(|(i, child)| {
                        let new_prefix = if prefix.is_empty() {
                            format!("[{i}]")
                        } else {
                            format!("{prefix}[{i}]")
                        };
                        extract_path(child, &segments[1..], &new_prefix)
                    })
                    .collect()
            } else {
                vec![]
            }
        }
    }
}

/// Generate a timestamped cache filename using epoch seconds.
fn generate_cache_filename() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("output_{secs}.json")
}

/// Write scan results in the requested format.
///
/// - `Json`: Pretty-printed JSON to stdout. When `output_file` is set, also
///   writes to the file.
/// - `Markdown`: Writes full JSON to `output_file` (or an auto-generated
///   `output_<epoch>.json`), then calls `to_markdown` with the cache path so
///   it can reference it. Markdown is printed to stdout.
///
/// When `fields` is non-empty, JSON output is filtered to only include values
/// matching the specified paths.
pub fn write_output<T: Serialize>(
    data: &T,
    format: &OutputFormat,
    output_file: Option<&Path>,
    fields: &[String],
    to_markdown: Option<impl FnOnce(&T, &str) -> String>,
) -> Result {
    match format {
        OutputFormat::Json => {
            let text = pretty_json(data, fields)?;
            if let Some(path) = output_file {
                let mut file = File::create(path)
                    .map_err(|e| Error::Io(format!("Failed to create {}: {e}", path.display())))?;
                writeln!(file, "{text}")
                    .map_err(|e| Error::Io(format!("Failed to write {}: {e}", path.display())))?;
                eprintln!("Results written to {}", path.display());
            }
            println!("{text}");
        }
        OutputFormat::Markdown => match to_markdown {
            Some(f) => {
                let generated = generate_cache_filename();
                let generated_path = Path::new(&generated);
                let cache_path = output_file.unwrap_or(generated_path);
                let json = serde_json::to_string_pretty(&data)
                    .map_err(|e| Error::Other(format!("JSON serialization for cache: {e}")))?;
                let mut file = File::create(cache_path).map_err(|e| {
                    Error::Io(format!(
                        "Failed to create cache {}: {e}",
                        cache_path.display()
                    ))
                })?;
                writeln!(file, "{json}").map_err(|e| {
                    Error::Io(format!(
                        "Failed to write cache {}: {e}",
                        cache_path.display()
                    ))
                })?;
                eprintln!("JSON cache written to {}", cache_path.display());
                let text = f(data, &cache_path.display().to_string());
                println!("{text}");
            }
            None => {
                let text = pretty_json(data, fields)?;
                println!("{text}");
            }
        },
    }
    Ok(())
}

fn pretty_json<T: Serialize>(data: &T, fields: &[String]) -> Result<String> {
    let value = serialize_value(data)?;
    let filtered = filter_fields(value, fields);
    serde_json::to_string_pretty(&filtered)
        .map_err(|e| Error::Other(format!("JSON serialization: {e}")))
}

fn serialize_value<T: Serialize>(data: &T) -> Result<serde_json::Value> {
    serde_json::to_value(data).map_err(|e| Error::Other(format!("JSON serialization: {e}")))
}

fn filter_fields(value: serde_json::Value, fields: &[String]) -> serde_json::Value {
    if fields.is_empty() {
        return value;
    }
    let paths: Vec<FieldPath> = fields.iter().map(|f| FieldPath::parse(f)).collect();
    match value {
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .into_iter()
                .map(|v| extract_fields(&v, &paths))
                .collect(),
        ),
        other => extract_fields(&other, &paths),
    }
}

fn extract_fields(value: &serde_json::Value, paths: &[FieldPath]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for path in paths {
        for (key, val) in extract_path(value, &path.segments, "") {
            map.insert(key, val);
        }
    }
    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_simple_key() {
        let path = FieldPath::parse("name");
        assert_eq!(path.segments, vec![PathSegment::Key("name".into())]);
    }

    #[test]
    fn parse_dot_path() {
        let path = FieldPath::parse("a.b.c");
        assert_eq!(
            path.segments,
            vec![
                PathSegment::Key("a".into()),
                PathSegment::Key("b".into()),
                PathSegment::Key("c".into()),
            ]
        );
    }

    #[test]
    fn parse_array_index() {
        let path = FieldPath::parse("items[0]");
        assert_eq!(
            path.segments,
            vec![PathSegment::Key("items".into()), PathSegment::Index(0)]
        );
    }

    #[test]
    fn parse_wildcard() {
        let path = FieldPath::parse("items[*]");
        assert_eq!(
            path.segments,
            vec![PathSegment::Key("items".into()), PathSegment::Wildcard]
        );
    }

    #[test]
    fn parse_combined() {
        let path = FieldPath::parse("results[*].response.status");
        assert_eq!(
            path.segments,
            vec![
                PathSegment::Key("results".into()),
                PathSegment::Wildcard,
                PathSegment::Key("response".into()),
                PathSegment::Key("status".into()),
            ]
        );
    }

    #[test]
    fn filter_empty_fields_no_change() {
        let value = json!({"a": 1, "b": 2});
        let result = filter_fields(value.clone(), &[]);
        assert_eq!(result, value);
    }

    #[test]
    fn filter_top_level() {
        let value = json!({"name": "x", "age": 1});
        let fields = vec!["name".to_string()];
        let result = filter_fields(value, &fields);
        assert_eq!(result, json!({"name": "x"}));
    }

    #[test]
    fn filter_nested_fields() {
        let value = json!({"resp": {"status": 200, "body": "ok"}, "id": 1});
        let fields = vec!["resp.status".to_string()];
        let result = filter_fields(value, &fields);
        assert_eq!(result, json!({"resp.status": 200}));
    }

    #[test]
    fn filter_array_of_objects() {
        let value = json!([
            {"name": "a", "resp": {"status": 200}},
            {"name": "b", "resp": {"status": 404}},
        ]);
        let fields = vec!["resp.status".to_string()];
        let result = filter_fields(value, &fields);
        assert_eq!(result, json!([{"resp.status": 200}, {"resp.status": 404}]));
    }

    #[test]
    fn extract_wildcard() {
        let value = json!({"items": [{"id": 1}, {"id": 2}]});
        let path = FieldPath::parse("items[*].id");
        let results = extract_path(&value, &path.segments, "");
        assert_eq!(
            results,
            vec![
                ("items[0].id".into(), json!(1)),
                ("items[1].id".into(), json!(2)),
            ]
        );
    }
}
