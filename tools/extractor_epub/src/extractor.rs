use crate::cli::ImageMode;
use crate::epub_reader;
use crate::models::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct ExtractionResult {
    pub output: BookOutput,
    /// Map from image filename to binary data (only used in Files mode)
    pub image_data: HashMap<String, Vec<u8>>,
}

pub fn extract_single(
    epub_path: &Path,
    image_mode: ImageMode,
) -> utl_helpers::Result<ExtractionResult> {
    let start = Instant::now();
    let source_file = epub_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown.epub".to_string());

    tracing::info!("Parsing EPUB: {}", epub_path.display());

    let epub = epub_reader::read_epub(epub_path, image_mode)?;

    let chapters: Vec<Chapter> = epub
        .chapters
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let title = epub
                .toc_labels
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Chapter {}", idx + 1));

            Chapter {
                title,
                index: idx,
                paragraphs: raw.paragraphs,
            }
        })
        .collect();

    // Book-level manifest: deduplicated image refs without base64 data
    let mut seen = std::collections::HashSet::new();
    let images: Vec<ImageRef> = chapters
        .iter()
        .flat_map(|ch| ch.paragraphs.iter())
        .flat_map(|p| p.images.iter())
        .filter(|img| seen.insert(img.id.clone()))
        .map(|img| ImageRef {
            path: img.path.clone(),
            id: img.id.clone(),
            media_type: img.media_type.clone(),
            data: None,
        })
        .collect();

    let duration = start.elapsed();
    let timestamp = format_iso8601();

    let extraction = ExtractionMetadata {
        extracted_at: timestamp,
        extraction_duration_ms: duration.as_millis() as u64,
        chapter_count: chapters.len(),
        image_count: images.len(),
        source_file,
    };

    tracing::info!(
        "Extracted {} chapters, {} images in {}ms",
        chapters.len(),
        images.len(),
        duration.as_millis()
    );

    Ok(ExtractionResult {
        output: BookOutput {
            book: Book { chapters, images },
            metadata: epub.metadata,
            extraction,
        },
        image_data: epub.image_file_data,
    })
}

/// Format current time as ISO 8601 (UTC).
fn format_iso8601() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Civil date from days since Unix epoch.
/// Howard Hinnant's algorithm: http://howardhinnant.github.io/date_algorithms.html
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    days += 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
