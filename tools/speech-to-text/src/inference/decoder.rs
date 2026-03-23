use super::language::token_id;
use super::WhisperModel;
use crate::models::Segment;
use candle_transformers::models::whisper;
use local_inference_helpers::candle_core::{Device, IndexOp, Tensor, D};
use local_inference_helpers::candle_nn;
use tokenizers::Tokenizer;

const MAX_TOKENS: usize = 448;
/// Whisper timestamp tokens start at this offset in the vocabulary.
/// Each token represents 0.02 seconds. Token <|0.00|> is at TIMESTAMP_BEGIN.
const TIMESTAMP_BEGIN: u32 = 50364;

pub struct DecoderOutput {
    pub tokens: Vec<u32>,
    pub segments: Vec<Segment>,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
}

/// Decode one 30-second mel spectrogram chunk into text + segments.
///
/// Implements greedy decoding with temperature fallback per the Whisper paper.
pub fn decode_chunk(
    model: &mut WhisperModel,
    audio_features: &Tensor,
    tokenizer: &Tokenizer,
    language: &str,
    translate: bool,
    timestamps: bool,
    device: &Device,
    chunk_offset_secs: f64,
) -> anyhow::Result<DecoderOutput> {
    let sot = token_id(tokenizer, whisper::SOT_TOKEN)?;
    let eot = token_id(tokenizer, whisper::EOT_TOKEN)?;
    let transcribe = token_id(tokenizer, whisper::TRANSCRIBE_TOKEN)?;
    let translate_token = token_id(tokenizer, whisper::TRANSLATE_TOKEN)?;
    let no_timestamps_token = token_id(tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
    let lang_token = token_id(tokenizer, &super::language::language_token(language))?;

    let task_token = if translate {
        translate_token
    } else {
        transcribe
    };

    let mut initial_tokens = vec![sot, lang_token, task_token];
    if !timestamps {
        initial_tokens.push(no_timestamps_token);
    }

    // Temperature fallback loop
    for &temperature in &whisper::TEMPERATURES {
        model.reset_kv_cache();

        match try_decode(
            model,
            audio_features,
            &initial_tokens,
            temperature,
            tokenizer,
            device,
            eot,
            timestamps,
            chunk_offset_secs,
        ) {
            Ok(output) => {
                // Quality check: skip if low confidence or garbled output
                if output.avg_logprob < whisper::LOGPROB_THRESHOLD
                    || compression_ratio(&output.tokens, tokenizer) > whisper::COMPRESSION_RATIO_THRESHOLD
                {
                    tracing::debug!(
                        temperature,
                        avg_logprob = output.avg_logprob,
                        "quality check failed, retrying with higher temperature"
                    );
                    continue;
                }
                return Ok(output);
            }
            Err(e) => {
                tracing::warn!(temperature, error = %e, "decode attempt failed");
                continue;
            }
        }
    }

    // Final attempt with highest temperature — return whatever we get
    model.reset_kv_cache();
    try_decode(
        model,
        audio_features,
        &initial_tokens,
        1.0,
        tokenizer,
        device,
        eot,
        timestamps,
        chunk_offset_secs,
    )
}

fn try_decode(
    model: &mut WhisperModel,
    audio_features: &Tensor,
    initial_tokens: &[u32],
    temperature: f64,
    tokenizer: &Tokenizer,
    device: &Device,
    eot: u32,
    timestamps: bool,
    chunk_offset_secs: f64,
) -> anyhow::Result<DecoderOutput> {
    let mut tokens = Vec::with_capacity(initial_tokens.len() + MAX_TOKENS);
    tokens.extend_from_slice(initial_tokens);
    let mut sum_logprob: f64 = 0.0;
    let mut count: usize = 0;
    let mut no_speech_prob: f64 = 0.0;

    // Feed initial tokens through decoder
    let input = Tensor::new(initial_tokens, device)?.unsqueeze(0)?;
    let logits = model.decoder_forward(&input, audio_features, true)?;

    // Check no-speech probability from first decoder output
    {
        let first_logits = logits.i((0, 0))?;
        let probs = candle_nn::ops::softmax_last_dim(&first_logits.unsqueeze(0)?)?.squeeze(0)?;
        for token_str in &whisper::NO_SPEECH_TOKENS {
            if let Ok(id) = token_id(tokenizer, token_str) {
                let p: f32 = probs.i(id as usize)?.to_scalar()?;
                no_speech_prob = no_speech_prob.max(p as f64);
            }
        }
    }

    // Get logits for the last position
    let last_pos = initial_tokens.len() - 1;
    let mut next_logits = logits.i((0, last_pos))?;

    for _ in 0..MAX_TOKENS {
        let next_token = if temperature == 0.0 {
            // Greedy argmax
            next_logits.argmax(D::Minus1)?.to_scalar::<u32>()?
        } else {
            // Sample from softmax(logits / temperature)
            let scaled = (&next_logits / temperature)?;
            let probs = candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0)?)?.squeeze(0)?;
            sample_from_probs(&probs)?
        };

        if next_token == eot {
            break;
        }

        // Track log probability
        let log_probs =
            candle_nn::ops::log_softmax(&next_logits.unsqueeze(0)?, D::Minus1)?.squeeze(0)?;
        let token_logprob: f32 = log_probs.i(next_token as usize)?.to_scalar()?;
        sum_logprob += token_logprob as f64;
        count += 1;

        tokens.push(next_token);

        // The candle Whisper model has no self-attention KV cache, so we must
        // pass ALL accumulated tokens each step. Cross-attention KV cache is
        // reused (flush_kv_cache = false) to avoid re-encoding.
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.decoder_forward(&input, audio_features, false)?;
        let seq_len = tokens.len() - 1;
        next_logits = logits.i((0, seq_len))?;
    }

    let avg_logprob = if count > 0 {
        sum_logprob / count as f64
    } else {
        0.0
    };

    // Extract text tokens (skip initial tokens and timestamp tokens)
    let text_tokens: Vec<u32> = tokens[initial_tokens.len()..]
        .iter()
        .copied()
        .filter(|&t| t < TIMESTAMP_BEGIN)
        .collect();

    let text = tokenizer
        .decode(&text_tokens, true)
        .unwrap_or_default()
        .trim()
        .to_string();

    // Extract segments from timestamp tokens
    let segments = if timestamps {
        extract_segments(&tokens[initial_tokens.len()..], tokenizer, chunk_offset_secs)
    } else {
        vec![Segment {
            start: chunk_offset_secs,
            end: chunk_offset_secs + 30.0,
            text: text.clone(),
        }]
    };

    Ok(DecoderOutput {
        tokens,
        segments,
        avg_logprob,
        no_speech_prob,
    })
}

/// Extract segments from decoded tokens using timestamp token pairs.
fn extract_segments(tokens: &[u32], tokenizer: &Tokenizer, chunk_offset: f64) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current_start: Option<f64> = None;
    let mut current_text_tokens: Vec<u32> = Vec::new();

    for &token in tokens {
        if token >= TIMESTAMP_BEGIN {
            let time = (token - TIMESTAMP_BEGIN) as f64 * 0.02 + chunk_offset;

            if let Some(start) = current_start {
                // This is an end timestamp — emit segment
                let text = tokenizer
                    .decode(&current_text_tokens, true)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if !text.is_empty() {
                    segments.push(Segment {
                        start,
                        end: time,
                        text,
                    });
                }
                current_start = None;
                current_text_tokens.clear();
            } else {
                // This is a start timestamp
                current_start = Some(time);
            }
        } else if current_start.is_some() {
            current_text_tokens.push(token);
        }
    }

    // Flush remaining tokens if we have a start but no end
    if let Some(start) = current_start {
        let text = tokenizer
            .decode(&current_text_tokens, true)
            .unwrap_or_default()
            .trim()
            .to_string();
        if !text.is_empty() {
            segments.push(Segment {
                start,
                end: start + 30.0,
                text,
            });
        }
    }

    segments
}

/// Compression ratio of decoded tokens as used in the Whisper paper:
/// len(text_bytes) / len(zlib_compressed_bytes).
/// We approximate with token count / unique token count, which correlates
/// with repetitiveness (the actual purpose of this check).
fn compression_ratio(tokens: &[u32], tokenizer: &Tokenizer) -> f64 {
    let text_tokens: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|&t| t < TIMESTAMP_BEGIN)
        .collect();
    let text = tokenizer.decode(&text_tokens, true).unwrap_or_default();
    if text.is_empty() {
        return 0.0;
    }
    let bytes = text.as_bytes();
    let mut unique = [false; 256];
    for &b in bytes {
        unique[b as usize] = true;
    }
    let unique_count = unique.iter().filter(|&&b| b).count();
    bytes.len() as f64 / unique_count.max(1) as f64
}

/// Sample a token from a probability distribution.
fn sample_from_probs(probs: &Tensor) -> anyhow::Result<u32> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let mut rng = rand::rng();
    let dist = rand::distr::weighted::WeightedIndex::new(&probs_vec)?;
    use rand::prelude::Distribution;
    Ok(dist.sample(&mut rng) as u32)
}

/// Get the last timestamp in seconds from decoded tokens, if any.
pub fn last_timestamp(tokens: &[u32]) -> Option<f64> {
    tokens
        .iter()
        .rev()
        .find(|&&t| t >= TIMESTAMP_BEGIN)
        .map(|&t| (t - TIMESTAMP_BEGIN) as f64 * 0.02)
}
