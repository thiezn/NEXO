use crate::openai::model::OpenAiServerControl;
use crate::openai::speech::OpenAiListenModel;

pub fn build_whisper_openai_model<S>(
    name: String,
    request_model_id: String,
    memory_bytes: u64,
    server: S,
    base_url: &str,
) -> OpenAiListenModel<S>
where
    S: OpenAiServerControl,
{
    OpenAiListenModel::new(
        name,
        "whisper",
        request_model_id,
        memory_bytes,
        server,
        base_url,
    )
}
