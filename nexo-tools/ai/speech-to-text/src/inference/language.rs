use super::WhisperModel;
use local_inference_helpers::candle_core::{Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

/// Whisper language tokens and their ISO 639-1 codes.
const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// Detect spoken language from audio features.
///
/// Feeds `[SOT]` to the decoder and inspects logits for language tokens.
/// Returns (language_code, probability).
pub fn detect_language(
    model: &mut WhisperModel,
    audio_features: &Tensor,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(String, f64)> {
    let sot_token = token_id(tokenizer, "<|startoftranscript|>")?;

    let tokens = Tensor::new(&[sot_token], device)?.unsqueeze(0)?;
    let logits = model.decoder_forward(&tokens, audio_features, true)?;

    // logits shape: [1, 1, vocab_size] — take last position
    let logits = logits.i((0, 0))?;

    // Collect language token IDs and their logits
    let mut lang_logits: Vec<(String, f32)> = Vec::new();
    for &(code, _name) in LANGUAGES {
        let token_str = format!("<|{code}|>");
        if let Ok(id) = token_id(tokenizer, &token_str) {
            let logit: f32 = logits.i(id as usize)?.to_scalar()?;
            lang_logits.push((code.to_string(), logit));
        }
    }

    if lang_logits.is_empty() {
        return Ok(("en".to_string(), 0.0));
    }

    // Softmax over language logits
    let max_logit = lang_logits
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = lang_logits.iter().map(|(_, l)| (l - max_logit).exp()).sum();

    let mut best_code = "en".to_string();
    let mut best_prob: f32 = 0.0;
    for (code, logit) in &lang_logits {
        let prob = (logit - max_logit).exp() / exp_sum;
        if prob > best_prob {
            best_prob = prob;
            best_code = code.clone();
        }
    }

    tracing::info!(language = %best_code, probability = %best_prob, "detected language");
    Ok((best_code, best_prob as f64))
}

/// Look up the token ID for a special token string.
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("token not found in vocabulary: {token}"))
}

/// Get the language token string for a language code.
pub fn language_token(code: &str) -> String {
    format!("<|{code}|>")
}
