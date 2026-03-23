pub fn system_prompt() -> &'static str {
    "You are an image captioning assistant for LoRA training datasets. \
     Describe the image concisely but thoroughly. Include the visual style, \
     colors, composition, subjects, and any notable details. \
     Write a single paragraph, no bullet points. \
     Do not start with 'This image shows' or 'The image depicts' or similar phrasing."
}

pub fn user_prompt() -> &'static str {
    "Describe this image for use as a LoRA training caption."
}
