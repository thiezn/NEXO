pub mod json;
pub mod plain;
pub mod srt;
pub mod vtt;

use crate::models::TranscriptionResult;

/// Output format for transcription results.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Text,
    Srt,
    Vtt,
    Json,
}

/// Format a transcription result according to the requested format.
pub fn format_output(result: &TranscriptionResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Text => plain::format(result),
        OutputFormat::Srt => srt::format(result),
        OutputFormat::Vtt => vtt::format(result),
        OutputFormat::Json => json::format(result),
    }
}

/// Format seconds as HH:MM:SS{sep}mmm timecode.
fn format_timecode(seconds: f64, ms_separator: char) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = total_secs / 3600;
    format!("{h:02}:{m:02}:{s:02}{ms_separator}{ms:03}")
}
