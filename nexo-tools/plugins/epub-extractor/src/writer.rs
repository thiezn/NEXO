use crate::cli::ImageMode;
use crate::extractor::ExtractionResult;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

pub fn book_output_dir(base_dir: &Path, result: &ExtractionResult) -> PathBuf {
    let raw_title = result
        .output
        .metadata
        .title
        .as_deref()
        .unwrap_or("unknown_book");

    let folder_name: String = raw_title
        .chars()
        .map(|c| if c == ' ' { '_' } else { c })
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .flat_map(|c| c.to_lowercase())
        .collect();

    let folder_name = if folder_name.is_empty() {
        "unknown_book".to_string()
    } else {
        folder_name
    };

    base_dir.join(folder_name)
}

pub fn write_result(
    base_dir: &Path,
    result: &ExtractionResult,
    image_mode: ImageMode,
) -> utl_helpers::Result<PathBuf> {
    let output_dir = book_output_dir(base_dir, result);

    // Remove existing output directory if it exists
    if output_dir.exists() {
        tracing::debug!("Removing existing output dir: {}", output_dir.display());
        std::fs::remove_dir_all(&output_dir).map_err(|e| {
            utl_helpers::Error::Io(format!(
                "Failed to remove existing dir '{}': {e}",
                output_dir.display()
            ))
        })?;
    }

    std::fs::create_dir_all(&output_dir).map_err(|e| {
        utl_helpers::Error::Io(format!(
            "Failed to create output dir '{}': {e}",
            output_dir.display()
        ))
    })?;

    let json_path = output_dir.join("book.json");
    let file = std::fs::File::create(&json_path).map_err(|e| {
        utl_helpers::Error::Io(format!(
            "Failed to create '{}': {e}",
            json_path.display()
        ))
    })?;
    serde_json::to_writer_pretty(BufWriter::new(file), &result.output)
        .map_err(|e| utl_helpers::Error::Other(format!("JSON serialization: {e}")))?;

    tracing::debug!("Wrote {}", json_path.display());

    if image_mode == ImageMode::Files && !result.image_data.is_empty() {
        let images_dir = output_dir.join("images");
        std::fs::create_dir_all(&images_dir).map_err(|e| {
            utl_helpers::Error::Io(format!(
                "Failed to create images dir '{}': {e}",
                images_dir.display()
            ))
        })?;

        for (filename, data) in &result.image_data {
            let img_path = images_dir.join(filename);
            std::fs::write(&img_path, data).map_err(|e| {
                utl_helpers::Error::Io(format!(
                    "Failed to write image '{}': {e}",
                    img_path.display()
                ))
            })?;
        }

        tracing::debug!(
            "Wrote {} images to {}",
            result.image_data.len(),
            images_dir.display()
        );
    }

    Ok(output_dir)
}
