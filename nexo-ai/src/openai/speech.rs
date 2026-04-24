use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::api::model_traits::{ListenModel, ModelInfo, TalkModel};
use crate::api::types::{
    ListenRequest, ListenResponse, ModelCategory, TalkRequest, TalkResponse, TranscriptionSegment,
};
use crate::audio::{AudioBuffer, encode_wav, load_bytes};

use super::client::OpenAiClient;
use super::model::OpenAiServerControl;
use super::protocol::{OpenAiSpeechRequest, OpenAiTranscriptionRequest};

const LISTEN_CATEGORIES: &[ModelCategory] = &[ModelCategory::Listen];
const TALK_CATEGORIES: &[ModelCategory] = &[ModelCategory::Talk];

pub type SpeechRequestBuilder = fn(&str, &TalkRequest) -> OpenAiSpeechRequest;

pub struct OpenAiRemoteModelCore<S = ()> {
    name: String,
    family: &'static str,
    request_model_id: String,
    memory_bytes: u64,
    server: S,
    client: OpenAiClient,
    loaded: bool,
}

impl<S> OpenAiRemoteModelCore<S>
where
    S: OpenAiServerControl,
{
    pub fn new(
        name: String,
        family: &'static str,
        request_model_id: String,
        memory_bytes: u64,
        server: S,
        base_url: &str,
    ) -> Self {
        Self {
            name,
            family,
            request_model_id,
            memory_bytes,
            server,
            client: OpenAiClient::new(base_url),
            loaded: false,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn family(&self) -> &str {
        self.family
    }

    pub fn memory_estimate_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn request_model_id(&self) -> &str {
        &self.request_model_id
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.server.ensure_running()?;
        self.loaded = true;
        Ok(())
    }

    pub fn unload(&mut self) {
        if !self.loaded {
            return;
        }
        let _ = self.server.unload_model(&self.request_model_id);
        self.loaded = false;
    }

    pub fn ensure_loaded(&self) -> Result<()> {
        if !self.loaded {
            bail!("model '{}' not loaded", self.name);
        }
        Ok(())
    }

    pub fn client(&self) -> &OpenAiClient {
        &self.client
    }

    pub fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
    }
}

pub struct OpenAiListenModel<S = ()> {
    core: OpenAiRemoteModelCore<S>,
}

impl<S> OpenAiListenModel<S>
where
    S: OpenAiServerControl,
{
    pub fn new(
        name: String,
        family: &'static str,
        request_model_id: String,
        memory_bytes: u64,
        server: S,
        base_url: &str,
    ) -> Self {
        Self {
            core: OpenAiRemoteModelCore::new(
                name,
                family,
                request_model_id,
                memory_bytes,
                server,
                base_url,
            ),
        }
    }
}

impl<S> ModelInfo for OpenAiListenModel<S>
where
    S: OpenAiServerControl,
{
    fn name(&self) -> &str {
        self.core.name()
    }

    fn family(&self) -> &str {
        self.core.family()
    }

    fn categories(&self) -> &[ModelCategory] {
        LISTEN_CATEGORIES
    }

    fn memory_estimate_bytes(&self) -> u64 {
        self.core.memory_estimate_bytes()
    }

    fn is_loaded(&self) -> bool {
        self.core.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        self.core.load()
    }

    fn unload(&mut self) {
        self.core.unload();
    }

    fn as_listen(&mut self) -> Option<&mut dyn ListenModel> {
        Some(self)
    }
}

impl<S> ListenModel for OpenAiListenModel<S>
where
    S: OpenAiServerControl,
{
    fn transcribe(&mut self, request: &ListenRequest) -> Result<ListenResponse> {
        self.core.ensure_loaded()?;

        let wav = encode_wav(&AudioBuffer::new(
            request.pcm_samples.clone(),
            request.sample_rate,
            1,
        ))
        .context("failed to encode transcription input as WAV")?;

        let wire = OpenAiTranscriptionRequest {
            model: self.core.request_model_id().to_string(),
            language: request.language.clone(),
            verbose: true,
            max_tokens: Some(1024),
            stream: false,
            context: None,
            text: None,
        };

        let start = Instant::now();
        let body = OpenAiRemoteModelCore::<S>::block_on(self.core.client().transcribe_audio(
            &wire,
            &wav,
            "audio.wav",
            "audio/wav",
        ))?;

        parse_transcription_ndjson(
            &body,
            request.language.clone(),
            start.elapsed().as_millis() as u64,
        )
    }
}

pub struct OpenAiTalkModel<S = ()> {
    core: OpenAiRemoteModelCore<S>,
    build_request: SpeechRequestBuilder,
}

impl<S> OpenAiTalkModel<S>
where
    S: OpenAiServerControl,
{
    pub fn new(
        name: String,
        family: &'static str,
        request_model_id: String,
        memory_bytes: u64,
        server: S,
        base_url: &str,
        build_request: SpeechRequestBuilder,
    ) -> Self {
        Self {
            core: OpenAiRemoteModelCore::new(
                name,
                family,
                request_model_id,
                memory_bytes,
                server,
                base_url,
            ),
            build_request,
        }
    }
}

impl<S> ModelInfo for OpenAiTalkModel<S>
where
    S: OpenAiServerControl,
{
    fn name(&self) -> &str {
        self.core.name()
    }

    fn family(&self) -> &str {
        self.core.family()
    }

    fn categories(&self) -> &[ModelCategory] {
        TALK_CATEGORIES
    }

    fn memory_estimate_bytes(&self) -> u64 {
        self.core.memory_estimate_bytes()
    }

    fn is_loaded(&self) -> bool {
        self.core.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        self.core.load()
    }

    fn unload(&mut self) {
        self.core.unload();
    }

    fn as_talk(&mut self) -> Option<&mut dyn TalkModel> {
        Some(self)
    }
}

impl<S> TalkModel for OpenAiTalkModel<S>
where
    S: OpenAiServerControl,
{
    fn synthesize(&mut self, request: &TalkRequest) -> Result<TalkResponse> {
        self.core.ensure_loaded()?;

        let wire = (self.build_request)(self.core.request_model_id(), request);
        let start = Instant::now();
        let body =
            OpenAiRemoteModelCore::<S>::block_on(self.core.client().synthesize_speech(&wire))?;
        decode_speech_response(&body, start.elapsed().as_millis() as u64)
    }
}

pub fn decode_speech_response(body: &[u8], inference_time_ms: u64) -> Result<TalkResponse> {
    let audio = load_bytes(body).context("failed to decode synthesized audio response")?;
    let audio = audio.to_mono();
    Ok(TalkResponse {
        pcm_samples: audio.samples,
        sample_rate: audio.sample_rate,
        inference_time_ms,
    })
}

pub fn parse_transcription_ndjson(
    body: &[u8],
    requested_language: Option<String>,
    inference_time_ms: u64,
) -> Result<ListenResponse> {
    let text_body =
        String::from_utf8(body.to_vec()).context("transcription response was not valid UTF-8")?;
    let mut full_text = String::new();
    let mut language = requested_language;
    let mut segments = Vec::new();

    for line in text_body.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("failed to parse transcription NDJSON line: {line}"))?;
        merge_transcription_value(&value, &mut full_text, &mut segments, &mut language);
    }

    if full_text.is_empty() && !segments.is_empty() {
        full_text = segments
            .iter()
            .map(|segment| segment.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
    }

    Ok(ListenResponse {
        text: full_text.trim().to_string(),
        segments,
        language,
        inference_time_ms,
    })
}

fn merge_transcription_value(
    value: &Value,
    full_text: &mut String,
    segments: &mut Vec<TranscriptionSegment>,
    language: &mut Option<String>,
) {
    if let Some(lang) = value.get("language").and_then(Value::as_str)
        && language.is_none()
    {
        *language = Some(lang.to_string());
    }

    if let Some(accumulated) = value.get("accumulated").and_then(Value::as_str) {
        *full_text = accumulated.to_string();
    }

    if let Some(text) = value.get("text").and_then(Value::as_str) {
        if value.get("start").is_some() || value.get("end").is_some() {
            segments.push(TranscriptionSegment {
                text: text.trim().to_string(),
                start_ms: numeric_to_millis(value.get("start")),
                end_ms: numeric_to_millis(value.get("end")),
            });
        } else if full_text.is_empty() || text.len() > full_text.len() {
            *full_text = text.to_string();
        }
    }

    if let Some(items) = value.get("segments").and_then(Value::as_array) {
        for item in items {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                segments.push(TranscriptionSegment {
                    text: text.trim().to_string(),
                    start_ms: numeric_to_millis(
                        item.get("start").or_else(|| item.get("start_time")),
                    ),
                    end_ms: numeric_to_millis(item.get("end").or_else(|| item.get("end_time"))),
                });
            }
        }
    }
}

fn numeric_to_millis(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(number)) => number
            .as_f64()
            .map(|value| (value * 1000.0).round().max(0.0) as u64)
            .unwrap_or(0),
        _ => 0,
    }
}
