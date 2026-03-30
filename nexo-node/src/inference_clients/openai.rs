use std::time::Instant;

use anyhow::{bail, Context};
use base64::Engine;
use serde::{Deserialize, Serialize};

use super::base::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ImageAnalysisRequest, ImageAnalysisResponse,
    TalkRequest, TalkResponse, ToolCall, ToolCallRequest, ToolCallResponse,
};

#[derive(Serialize)]
struct OaiRequest {
    model: String,
    messages: Vec<OaiMessage>,
    max_tokens: usize,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
struct OaiMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct OaiResponse {
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiChoiceMessage,
}

#[derive(Deserialize)]
struct OaiChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Deserialize)]
struct OaiToolCall {
    function: OaiToolCallFunction,
}

#[derive(Deserialize)]
struct OaiToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OaiUsage {
    #[serde(default)]
    completion_tokens: usize,
}

#[derive(Serialize)]
struct TtsRequest {
    model: String,
    input: String,
    voice: String,
    response_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruct: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

struct CompletionResult {
    choice: OaiChoiceMessage,
    tokens_generated: usize,
    inference_time_ms: u64,
}

async fn post_chat_completion(
    http: &reqwest::Client,
    base_url: &str,
    body: &OaiRequest,
    label: &str,
) -> anyhow::Result<CompletionResult> {
    let start = Instant::now();

    let resp: OaiResponse = http
        .post(format!("{base_url}/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .with_context(|| format!("failed to reach {label}"))?
        .error_for_status()
        .with_context(|| format!("{label} returned an error"))?
        .json()
        .await
        .with_context(|| format!("failed to parse {label} response"))?;

    let elapsed = start.elapsed();

    let choice = resp
        .choices
        .into_iter()
        .next()
        .with_context(|| format!("{label} returned no choices"))?;

    Ok(CompletionResult {
        choice: choice.message,
        tokens_generated: resp.usage.map_or(0, |u| u.completion_tokens),
        inference_time_ms: elapsed.as_millis() as u64,
    })
}

fn role_str(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

fn to_oai_messages(messages: &[ChatMessage]) -> Vec<OaiMessage> {
    messages
        .iter()
        .map(|m| OaiMessage {
            role: role_str(m.role).to_owned(),
            content: serde_json::Value::String(m.content.clone()),
        })
        .collect()
}

fn sniff_mime(data: &[u8]) -> &'static str {
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        "image/png"
    } else if data.starts_with(&[0xFF, 0xD8]) {
        "image/jpeg"
    } else if data.starts_with(b"GIF8") {
        "image/gif"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

pub(super) async fn chat(
    http: &reqwest::Client,
    base_url: &str,
    req: ChatRequest,
) -> anyhow::Result<ChatResponse> {
    let body = OaiRequest {
        model: "default".into(),
        messages: to_oai_messages(&req.messages),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: Some(req.top_p),
        tools: None,
    };

    let result = post_chat_completion(http, base_url, &body, "llama-server").await?;

    Ok(ChatResponse {
        text: result.choice.content.unwrap_or_default(),
        tokens_generated: result.tokens_generated,
        inference_time_ms: result.inference_time_ms,
    })
}

pub(super) async fn tool_call(
    http: &reqwest::Client,
    base_url: &str,
    req: ToolCallRequest,
) -> anyhow::Result<ToolCallResponse> {
    let body = OaiRequest {
        model: "default".into(),
        messages: to_oai_messages(&req.messages),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: None,
        tools: Some(req.tools),
    };

    let result = post_chat_completion(http, base_url, &body, "llama-server").await?;

    let tool_calls = match result.choice.tool_calls {
        Some(calls) => calls
            .into_iter()
            .map(|tc| {
                let arguments = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::String(tc.function.arguments));
                ToolCall {
                    name: tc.function.name,
                    arguments,
                }
            })
            .collect(),
        None => vec![],
    };

    Ok(ToolCallResponse {
        tool_calls,
        reasoning: result.choice.content,
        tokens_generated: result.tokens_generated,
        inference_time_ms: result.inference_time_ms,
    })
}

pub(super) async fn analyze_image(
    http: &reqwest::Client,
    base_url: &str,
    req: ImageAnalysisRequest,
) -> anyhow::Result<ImageAnalysisResponse> {
    let mime = sniff_mime(&req.image_data);
    let prefix = format!("data:{mime};base64,");
    let mut data_uri = String::with_capacity(prefix.len() + req.image_data.len() * 4 / 3 + 4);
    data_uri.push_str(&prefix);
    base64::engine::general_purpose::STANDARD.encode_string(&req.image_data, &mut data_uri);

    let content = serde_json::json!([
        { "type": "text", "text": req.prompt },
        { "type": "image_url", "image_url": { "url": data_uri } }
    ]);

    let body = OaiRequest {
        model: "default".into(),
        messages: vec![OaiMessage {
            role: role_str(ChatRole::User).to_owned(),
            content,
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: None,
        tools: None,
    };

    let result = post_chat_completion(http, base_url, &body, "vllm-mlx").await?;

    Ok(ImageAnalysisResponse {
        text: result.choice.content.unwrap_or_default(),
        tokens_generated: result.tokens_generated,
        inference_time_ms: result.inference_time_ms,
    })
}

pub(super) async fn talk(
    http: &reqwest::Client,
    base_url: &str,
    req: TalkRequest,
) -> anyhow::Result<TalkResponse> {
    let body = TtsRequest {
        model: "tts-1".into(),
        input: req.text,
        voice: req.voice,
        response_format: "wav".into(),
        instruct: req.instruct,
        language: req.language,
        speed: req.speed,
    };

    let start = Instant::now();

    let wav_bytes = http
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
        .await
        .context("failed to reach mlx-tts-server")?
        .error_for_status()
        .context("mlx-tts-server returned an error")?
        .bytes()
        .await
        .context("failed to read mlx-tts-server response body")?;

    let elapsed = start.elapsed();

    let (pcm_samples, sample_rate) = parse_wav(&wav_bytes)?;

    Ok(TalkResponse {
        pcm_samples,
        sample_rate,
        inference_time_ms: elapsed.as_millis() as u64,
    })
}

/// Supports 16-bit PCM and 32-bit float WAV.
fn parse_wav(data: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    if data.len() < 44 {
        bail!("WAV data too short ({} bytes)", data.len());
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("not a valid WAV file");
    }

    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    // Find the "data" chunk — may not start at byte 44 if extra chunks are present.
    let mut offset = 12;
    let pcm_data = loop {
        if offset + 8 > data.len() {
            bail!("could not find data chunk in WAV");
        }
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        if chunk_id == b"data" {
            let start = offset + 8;
            let end = (start + chunk_size).min(data.len());
            break &data[start..end];
        }
        offset += 8 + chunk_size;
    };

    let samples: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => pcm_data
            .chunks_exact(2)
            .map(|c| {
                let sample = i16::from_le_bytes([c[0], c[1]]);
                sample as f32 / 32768.0
            })
            .collect(),
        (3, 32) => pcm_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => bail!(
            "unsupported WAV format: audio_format={audio_format}, bits_per_sample={bits_per_sample}"
        ),
    };

    let mono = if num_channels == 2 {
        samples
            .chunks_exact(2)
            .map(|pair| (pair[0] + pair[1]) * 0.5)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

#[derive(Serialize)]
struct ModelManageRequest {
    model_id: String,
}

/// Request the llama-server to load a model into VRAM.
pub(super) async fn load_model(
    http: &reqwest::Client,
    base_url: &str,
    model_id: &str,
) -> anyhow::Result<()> {
    http.post(format!("{base_url}/v1/models/load"))
        .json(&ModelManageRequest {
            model_id: model_id.to_string(),
        })
        .send()
        .await
        .context("failed to reach llama-server for model load")?
        .error_for_status()
        .context("llama-server model load failed")?;
    Ok(())
}

/// Request the llama-server to unload a model from VRAM.
pub(super) async fn unload_model(
    http: &reqwest::Client,
    base_url: &str,
    model_id: &str,
) -> anyhow::Result<()> {
    http.post(format!("{base_url}/v1/models/unload"))
        .json(&ModelManageRequest {
            model_id: model_id.to_string(),
        })
        .send()
        .await
        .context("failed to reach llama-server for model unload")?
        .error_for_status()
        .context("llama-server model unload failed")?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn role_str_maps_correctly() {
        assert_eq!(role_str(ChatRole::System), "system");
        assert_eq!(role_str(ChatRole::User), "user");
        assert_eq!(role_str(ChatRole::Assistant), "assistant");
        assert_eq!(role_str(ChatRole::Tool), "tool");
    }

    #[test]
    fn to_oai_messages_converts() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        }];
        let oai = to_oai_messages(&msgs);
        assert_eq!(oai.len(), 1);
        assert_eq!(oai[0].role, "user");
        assert_eq!(oai[0].content, serde_json::Value::String("hello".into()));
    }

    #[test]
    fn oai_request_serializes_without_optional_fields() {
        let body = OaiRequest {
            model: "default".into(),
            messages: vec![OaiMessage {
                role: "user".into(),
                content: serde_json::Value::String("hi".into()),
            }],
            max_tokens: 100,
            temperature: 0.7,
            top_p: None,
            tools: None,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("top_p").is_none());
        assert!(json.get("tools").is_none());
        assert_eq!(json["model"], "default");
    }

    #[test]
    fn sniff_mime_detects_formats() {
        assert_eq!(sniff_mime(&[0x89, 0x50, 0x4E, 0x47, 0x0D]), "image/png");
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF]), "image/jpeg");
        assert_eq!(sniff_mime(b"GIF89a"), "image/gif");
        let mut webp = vec![0u8; 12];
        webp[0..4].copy_from_slice(b"RIFF");
        webp[8..12].copy_from_slice(b"WEBP");
        assert_eq!(sniff_mime(&webp), "image/webp");
        assert_eq!(sniff_mime(&[0x00, 0x01]), "application/octet-stream");
    }

    #[test]
    fn parse_wav_16bit_mono() {
        let sample_rate: u32 = 16000;
        let samples: [i16; 4] = [0, 16384, -16384, 32767];
        let data_size = (samples.len() * 2) as u32;

        let mut wav = Vec::with_capacity(44 + data_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for s in &samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }

        let (pcm, sr) = parse_wav(&wav).unwrap();
        assert_eq!(sr, 16000);
        assert_eq!(pcm.len(), 4);
        assert!((pcm[0]).abs() < 0.001);
        assert!((pcm[1] - 0.5).abs() < 0.01);
        assert!((pcm[2] + 0.5).abs() < 0.01);
    }

    #[test]
    fn parse_wav_rejects_short_data() {
        assert!(parse_wav(&[0u8; 10]).is_err());
    }

    #[test]
    fn parse_wav_rejects_non_wav() {
        let mut data = [0u8; 44];
        data[0..4].copy_from_slice(b"NOPE");
        assert!(parse_wav(&data).is_err());
    }
}
