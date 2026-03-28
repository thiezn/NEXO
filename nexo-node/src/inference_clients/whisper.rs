use std::time::Instant;

use anyhow::Context;

use super::base::{ListenRequest, ListenResponse, TranscriptionSegment};


fn pcm_f32_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i16_val.to_le_bytes());
    }

    buf
}


fn parse_segments(body: &serde_json::Value) -> Vec<TranscriptionSegment> {
    let Some(segments) = body.get("segments").and_then(|s| s.as_array()) else {
        return vec![];
    };

    segments
        .iter()
        .filter_map(|seg| {
            let text = seg.get("text")?.as_str()?.trim().to_string();

            // whisper.cpp uses t0/t1 (in centiseconds) or offsets/timestamps fields.
            // Try t0/t1 first, then fall back to start/end (in seconds).
            let (start_ms, end_ms) = if let (Some(t0), Some(t1)) =
                (seg.get("t0").and_then(|v| v.as_u64()), seg.get("t1").and_then(|v| v.as_u64()))
            {
                // t0/t1 are in centiseconds (10ms units)
                (t0 * 10, t1 * 10)
            } else if let (Some(start), Some(end)) = (
                seg.get("start").and_then(|v| v.as_f64()),
                seg.get("end").and_then(|v| v.as_f64()),
            ) {
                ((start * 1000.0) as u64, (end * 1000.0) as u64)
            } else {
                (0, 0)
            };

            Some(TranscriptionSegment {
                text,
                start_ms,
                end_ms,
            })
        })
        .collect()
}


pub(super) async fn transcribe(
    http: &reqwest::Client,
    base_url: &str,
    req: ListenRequest,
) -> anyhow::Result<ListenResponse> {
    let wav_bytes = pcm_f32_to_wav(&req.pcm_samples, req.sample_rate);

    let file_part = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("temperature", "0.0")
        .text("temperature_inc", "0.2")
        .text("response_format", "json");

    if let Some(ref lang) = req.language {
        form = form.text("language", lang.clone());
    }

    let start = Instant::now();

    let body: serde_json::Value = http
        .post(format!("{base_url}/inference"))
        .multipart(form)
        .send()
        .await
        .context("failed to reach whisper-server")?
        .error_for_status()
        .context("whisper-server returned an error")?
        .json()
        .await
        .context("failed to parse whisper-server response")?;

    let elapsed = start.elapsed();

    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();

    let segments = parse_segments(&body);
    let language = body.get("language").and_then(|v| v.as_str()).map(String::from);

    Ok(ListenResponse {
        text,
        segments,
        language,
        inference_time_ms: elapsed.as_millis() as u64,
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pcm_to_wav_produces_valid_header() {
        let samples = vec![0.0f32; 100];
        let wav = pcm_f32_to_wav(&samples, 16000);

        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // fmt chunk size = 16
        assert_eq!(u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]), 16);
        // audio format = 1 (PCM)
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
        // channels = 1
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        // sample rate = 16000
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16000
        );
        // bits per sample = 16
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
        // data chunk
        assert_eq!(&wav[36..40], b"data");
        // data size = 200 (100 samples * 2 bytes)
        assert_eq!(
            u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]),
            200
        );
        // total size
        assert_eq!(wav.len(), 44 + 200);
    }

    #[test]
    fn pcm_to_wav_clamps_values() {
        let samples = vec![2.0, -2.0];
        let wav = pcm_f32_to_wav(&samples, 16000);
        // First sample should be clamped to 32767
        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        assert_eq!(s0, 32767);
        // Second sample should be clamped to -32767 (since -1.0 * 32767 = -32767)
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(s1, -32767);
    }

    #[test]
    fn parse_segments_with_t0_t1() {
        let body = serde_json::json!({
            "text": "hello world",
            "segments": [
                { "text": " hello ", "t0": 0, "t1": 50 },
                { "text": " world ", "t0": 50, "t1": 100 }
            ]
        });
        let segs = parse_segments(&body);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "hello");
        assert_eq!(segs[0].start_ms, 0);
        assert_eq!(segs[0].end_ms, 500);
        assert_eq!(segs[1].text, "world");
        assert_eq!(segs[1].start_ms, 500);
        assert_eq!(segs[1].end_ms, 1000);
    }

    #[test]
    fn parse_segments_with_start_end() {
        let body = serde_json::json!({
            "text": "test",
            "segments": [
                { "text": "test", "start": 0.5, "end": 1.2 }
            ]
        });
        let segs = parse_segments(&body);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 500);
        assert_eq!(segs[0].end_ms, 1200);
    }

    #[test]
    fn parse_segments_empty_when_missing() {
        let body = serde_json::json!({ "text": "no segments" });
        assert!(parse_segments(&body).is_empty());
    }
}
