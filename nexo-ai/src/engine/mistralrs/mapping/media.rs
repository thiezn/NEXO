use crate::{Error, Result};
use mistralrs_core::AudioInput as MistralAudioInput;
use nexo_core::{ImageInput, MediaSource};

/// Maps an image input into the in-memory representation expected by Mistral.rs.
///
/// # Arguments
///
/// * `input` - The image input that should be decoded into an image buffer.
pub(crate) fn map_image_input(input: &ImageInput) -> Result<image::DynamicImage> {
    let bytes = media_source_bytes(&input.source, "image")?;
    image::load_from_memory(&bytes).map_err(|error| {
        nexo_core::Error::InvalidRequest {
            message: format!("invalid image input: {error}"),
        }
        .into()
    })
}

/// Maps an audio input into the audio payload expected by Mistral.rs.
///
/// # Arguments
///
/// * `input` - The audio input that should be decoded into a Mistral.rs audio payload.
pub(crate) fn map_audio_input(input: &nexo_core::AudioInput) -> Result<MistralAudioInput> {
    let bytes = media_source_bytes(&input.source, "audio")?;
    MistralAudioInput::from_bytes(&bytes).map_err(|error| {
        nexo_core::Error::InvalidRequest {
            message: format!("invalid audio input: {error}"),
        }
        .into()
    })
}

/// Resolves raw bytes from a transport-safe media source.
///
/// # Arguments
///
/// * `source` - The media source carrying bytes, base64, or a data URL.
/// * `part` - The human-readable media kind used in validation errors.
pub(crate) fn media_source_bytes(source: &MediaSource, part: &str) -> Result<Vec<u8>> {
    match source {
        MediaSource::Bytes(bytes) => Ok(bytes.clone()),
        MediaSource::Base64(encoded) => decode_base64_bytes(encoded, part),
        MediaSource::Url(url) => {
            if let Some(payload) = url.strip_prefix("data:")
                && let Some((_, base64_data)) = payload.split_once(";base64,")
            {
                return decode_base64_bytes(base64_data, part);
            }

            Err(Error::UnsupportedMessagePart {
                part: "non-data-url media source",
            })
        }
    }
}

/// Decodes a base64-encoded media payload into raw bytes.
///
/// # Arguments
///
/// * `encoded` - The base64 payload to decode.
/// * `part` - The human-readable media kind used in validation errors.
fn decode_base64_bytes(encoded: &str, part: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| {
            nexo_core::Error::InvalidRequest {
                message: format!("invalid {part} base64 payload: {error}"),
            }
            .into()
        })
}
