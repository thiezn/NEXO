use crate::models::TranscriptionResult;
use super::format_timecode;

pub fn format(result: &TranscriptionResult) -> String {
    let mut out = String::new();
    for (i, seg) in result.segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_timecode(seg.start, ','),
            format_timecode(seg.end, ',')
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}
