use crate::connect::ConnectParams;
use crate::error::ErrorPayload;
use crate::events::EventKind;
use crate::frame::Frame;
use crate::methods::Method;

/// Available schema sections for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum SchemaSection {
    All,
    Frames,
    Connect,
    Methods,
    Events,
    Errors,
}

/// Generate pretty-printed JSON Schema string for a section.
pub fn schema_json(section: SchemaSection) -> String {
    let schema = generate_schema(section);
    serde_json::to_string_pretty(&schema).unwrap_or_default()
}

/// Generate JSON Schema for a specific section or the full protocol.
pub fn generate_schema(section: SchemaSection) -> serde_json::Value {
    match section {
        SchemaSection::All => serde_json::json!({
            "frames": serde_json::to_value(schemars::schema_for!(Frame)).unwrap_or_default(),
            "connect": serde_json::to_value(schemars::schema_for!(ConnectParams)).unwrap_or_default(),
            "methods": serde_json::to_value(schemars::schema_for!(Method)).unwrap_or_default(),
            "events": serde_json::to_value(schemars::schema_for!(EventKind)).unwrap_or_default(),
            "errors": serde_json::to_value(schemars::schema_for!(ErrorPayload)).unwrap_or_default(),
        }),
        SchemaSection::Frames => {
            serde_json::to_value(schemars::schema_for!(Frame)).unwrap_or_default()
        }
        SchemaSection::Connect => {
            serde_json::to_value(schemars::schema_for!(ConnectParams)).unwrap_or_default()
        }
        SchemaSection::Methods => {
            serde_json::to_value(schemars::schema_for!(Method)).unwrap_or_default()
        }
        SchemaSection::Events => {
            serde_json::to_value(schemars::schema_for!(EventKind)).unwrap_or_default()
        }
        SchemaSection::Errors => {
            serde_json::to_value(schemars::schema_for!(ErrorPayload)).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn generate_all_schema_not_empty() {
        let schema = generate_schema(SchemaSection::All);
        assert!(schema.is_object());
        let obj = schema.as_object().unwrap();
        assert!(obj.contains_key("frames"));
        assert!(obj.contains_key("connect"));
        assert!(obj.contains_key("methods"));
        assert!(obj.contains_key("events"));
        assert!(obj.contains_key("errors"));
    }

    #[test]
    fn each_section_produces_valid_json_schema() {
        for section in [
            SchemaSection::Frames,
            SchemaSection::Connect,
            SchemaSection::Methods,
            SchemaSection::Events,
            SchemaSection::Errors,
        ] {
            let schema = generate_schema(section);
            assert!(
                schema.is_object(),
                "Section {section:?} should be an object"
            );
            let obj = schema.as_object().unwrap();
            // All JSON Schema objects should have a title or definitions
            assert!(
                obj.contains_key("title")
                    || obj.contains_key("$schema")
                    || obj.contains_key("oneOf")
                    || obj.contains_key("type")
                    || obj.contains_key("definitions"),
                "Section {section:?} should look like a JSON Schema: {obj:?}"
            );
        }
    }

    #[test]
    fn frames_schema_contains_request_response_event() {
        let schema = generate_schema(SchemaSection::Frames);
        let json_str = serde_json::to_string(&schema).unwrap();
        assert!(json_str.contains("Request") || json_str.contains("request"));
        assert!(json_str.contains("Response") || json_str.contains("response"));
        assert!(json_str.contains("Event") || json_str.contains("event"));
    }
}
