/// Build the system prompt that instructs the text LLM to convert
/// prose paragraphs into an image generation prompt.
pub(crate) fn build_system_prompt() -> String {
    "You are an expert at creating image generation prompts. \
     Given one or more paragraphs of text, distill the visual essence into a single, \
     detailed image generation prompt. Focus on visual elements: subjects, composition, \
     lighting, mood, colors, and style. \
     Return ONLY the image generation prompt text, nothing else. \
     Do not include any explanation, preamble, or formatting."
        .to_string()
}

/// Combine paragraphs into a single user message.
pub(crate) fn build_user_message(paragraphs: &[String]) -> String {
    paragraphs.join("\n\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_is_not_empty() {
        let prompt = build_system_prompt();
        assert!(!prompt.is_empty());
    }

    #[test]
    fn user_message_combines_paragraphs() {
        let paragraphs = vec![
            "First paragraph.".to_string(),
            "Second paragraph.".to_string(),
        ];
        let msg = build_user_message(&paragraphs);
        assert!(msg.contains("First paragraph."));
        assert!(msg.contains("Second paragraph."));
        assert!(msg.contains("\n\n"));
    }

    #[test]
    fn user_message_single_paragraph() {
        let paragraphs = vec!["Only one.".to_string()];
        let msg = build_user_message(&paragraphs);
        assert_eq!(msg, "Only one.");
    }

    #[test]
    fn user_message_empty() {
        let paragraphs: Vec<String> = vec![];
        let msg = build_user_message(&paragraphs);
        assert!(msg.is_empty());
    }
}
