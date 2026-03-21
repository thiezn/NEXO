use serde::Serialize;

use super::image::ImageRef;

#[derive(Debug, Serialize)]
pub struct Paragraph {
    pub text: Vec<String>,
    pub images: Vec<ImageRef>,
}
