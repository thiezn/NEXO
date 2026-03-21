use serde::Serialize;

use super::paragraph::Paragraph;

#[derive(Debug, Serialize)]
pub struct Chapter {
    pub title: String,
    pub index: usize,
    pub paragraphs: Vec<Paragraph>,
}
