use crate::models::TranscriptionResult;

pub fn format(result: &TranscriptionResult) -> String {
    result.text.clone()
}
