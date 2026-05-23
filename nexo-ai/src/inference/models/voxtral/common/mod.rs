use crate::api::types::TalkRequest;
use crate::inference::remote::openai::protocol::OpenAiSpeechRequest;

pub fn build_voxtral_speech_request(model_id: &str, request: &TalkRequest) -> OpenAiSpeechRequest {
    OpenAiSpeechRequest {
        model: model_id.to_string(),
        input: request.text.clone(),
        instruct: (!request.voice_description.trim().is_empty())
            .then(|| request.voice_description.clone()),
        voice: None,
        speed: Some(1.0),
        lang_code: None,
        temperature: Some(request.temperature),
        max_tokens: Some(request.max_tokens),
        response_format: "wav".to_string(),
        stream: false,
    }
}
