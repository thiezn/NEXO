use crate::inference::remote::openai::model::OpenAiServerControl;
use crate::inference::remote::openai::speech::OpenAiTalkModel;

use super::common::build_voxtral_speech_request;

pub fn build_voxtral_openai_model<S>(
    name: String,
    request_model_id: String,
    memory_bytes: u64,
    server: S,
    base_url: &str,
) -> OpenAiTalkModel<S>
where
    S: OpenAiServerControl,
{
    OpenAiTalkModel::new(
        name,
        "voxtral",
        request_model_id,
        memory_bytes,
        server,
        base_url,
        build_voxtral_speech_request,
    )
}
