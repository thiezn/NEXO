use serde::Serialize;

use super::chapter::Chapter;
use super::image::ImageRef;

#[derive(Debug, Serialize)]
pub struct BookMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub publisher: Option<String>,
    pub language: Option<String>,
    pub identifier: Option<String>,
    pub date: Option<String>,
    pub rights: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExtractionMetadata {
    pub extracted_at: String,
    pub extraction_duration_ms: u64,
    pub chapter_count: usize,
    pub image_count: usize,
    pub source_file: String,
}

#[derive(Debug, Serialize)]
pub struct Book {
    pub chapters: Vec<Chapter>,
    pub images: Vec<ImageRef>,
}

#[derive(Debug, Serialize)]
pub struct BookOutput {
    pub book: Book,
    pub metadata: BookMetadata,
    pub extraction: ExtractionMetadata,
}
