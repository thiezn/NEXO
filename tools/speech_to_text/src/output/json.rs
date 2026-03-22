use crate::models::TranscriptionResult;

pub fn format(result: &TranscriptionResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}
