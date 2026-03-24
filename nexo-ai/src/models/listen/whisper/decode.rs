use anyhow::Result;
use candle_transformers::models::whisper;

use crate::shared::types::TranscriptionSegment;

/// Resolved special token IDs for Whisper decoding.
pub struct WhisperTokens {
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub no_timestamps: u32,
    pub no_speech: Vec<u32>,
    /// First timestamp token ID — `<|0.00|>`.
    pub timestamp_base: u32,
}

impl WhisperTokens {
    pub fn new(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let resolve = |token: &str| -> Result<u32> {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| anyhow::anyhow!("token '{token}' not in vocabulary"))
        };

        let sot = resolve(whisper::SOT_TOKEN)?;
        let eot = resolve(whisper::EOT_TOKEN)?;
        let transcribe = resolve(whisper::TRANSCRIBE_TOKEN)?;
        let no_timestamps = resolve(whisper::NO_TIMESTAMPS_TOKEN)?;

        let no_speech: Vec<u32> = whisper::NO_SPEECH_TOKENS
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();

        let timestamp_base = resolve("<|0.00|>")?;

        Ok(Self {
            sot,
            eot,
            transcribe,
            no_timestamps,
            no_speech,
            timestamp_base,
        })
    }

    /// Build the initial decoder token sequence.
    ///
    /// Format: `[sot, language_id, transcribe]`
    /// If no language is specified, defaults to English.
    pub fn initial_tokens(&self, tokenizer: &tokenizers::Tokenizer, language: Option<&str>) -> Vec<u32> {
        let lang_token = language
            .and_then(|lang| {
                let token_str = format!("<|{lang}|>");
                tokenizer.token_to_id(&token_str)
            })
            .unwrap_or_else(|| {
                tokenizer.token_to_id("<|en|>").unwrap_or(self.sot + 1)
            });

        vec![self.sot, lang_token, self.transcribe]
    }

    /// Check whether a token ID is a timestamp token.
    pub fn is_timestamp(&self, token: u32) -> bool {
        token >= self.timestamp_base
    }

    /// Convert a timestamp token ID to milliseconds.
    ///
    /// Each timestamp token step is 20ms: `<|0.00|>` = 0ms, `<|0.02|>` = 20ms, etc.
    pub fn timestamp_to_ms(&self, token: u32) -> u64 {
        (token - self.timestamp_base) as u64 * 20
    }
}

/// Decode a sequence of token IDs into text and timestamp-delimited segments.
///
/// `chunk_offset_ms` is added to all segment timestamps to account for chunked audio.
pub fn decode_with_timestamps(
    tokens: &WhisperTokens,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    chunk_offset_ms: u64,
) -> (String, Vec<TranscriptionSegment>) {
    let mut segments = Vec::new();
    let mut full_text = String::new();

    let mut current_start_ms: Option<u64> = None;
    let mut current_tokens: Vec<u32> = Vec::new();

    for &id in token_ids {
        if id == tokens.eot {
            break;
        }

        if tokens.is_timestamp(id) {
            let ts_ms = tokens.timestamp_to_ms(id) + chunk_offset_ms;

            if let Some(start) = current_start_ms {
                // This is the end timestamp — flush segment
                if !current_tokens.is_empty() {
                    let text = tokenizer
                        .decode(&current_tokens, true)
                        .unwrap_or_default();
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() {
                        full_text.push_str(&trimmed);
                        full_text.push(' ');
                        segments.push(TranscriptionSegment {
                            text: trimmed,
                            start_ms: start,
                            end_ms: ts_ms,
                        });
                    }
                    current_tokens.clear();
                }
                current_start_ms = None;
            } else {
                // This is the start timestamp
                current_start_ms = Some(ts_ms);
            }
        } else {
            current_tokens.push(id);
        }
    }

    // Flush any remaining tokens without a closing timestamp
    if !current_tokens.is_empty() {
        let text = tokenizer
            .decode(&current_tokens, true)
            .unwrap_or_default();
        let trimmed = text.trim().to_string();
        if !trimmed.is_empty() {
            full_text.push_str(&trimmed);
        }
    }

    let full_text = full_text.trim().to_string();
    (full_text, segments)
}

/// Build a boolean mask indicating which tokens should be suppressed during decoding.
pub fn build_suppress_mask(suppress_tokens: &[u32], vocab_size: usize) -> Vec<bool> {
    let mut mask = vec![false; vocab_size];
    for &token in suppress_tokens {
        if (token as usize) < vocab_size {
            mask[token as usize] = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppress_mask_marks_correct_tokens() {
        let mask = build_suppress_mask(&[1, 5, 10], 20);
        assert_eq!(mask.len(), 20);
        assert!(mask[1]);
        assert!(mask[5]);
        assert!(mask[10]);
        assert!(!mask[0]);
        assert!(!mask[3]);
    }

    #[test]
    fn suppress_mask_ignores_out_of_range() {
        let mask = build_suppress_mask(&[100], 10);
        assert_eq!(mask.len(), 10);
        assert!(mask.iter().all(|&v| !v));
    }
}
