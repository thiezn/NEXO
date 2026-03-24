use std::borrow::Cow;

use anyhow::Result;

/// Normalize a Parler-TTS config.json to ensure its `audio_encoder` section
/// matches the schema expected by candle-transformers' DAC `Config` struct.
///
/// Parler-TTS mini v1.1 uses a newer DAC config format with `n_codebooks`,
/// `downsampling_ratios`, `hop_length`, etc. instead of the older format's
/// `num_codebooks`, `frame_rate`, `latent_dim`, `model_bitrate`.
pub fn normalize_config(config_str: &str) -> Result<Cow<'_, str>> {
    if config_str.contains("\"num_codebooks\"")
        && config_str.contains("\"frame_rate\"")
        && config_str.contains("\"latent_dim\"")
        && config_str.contains("\"model_bitrate\"")
    {
        return Ok(Cow::Borrowed(config_str));
    }

    let mut root: serde_json::Value = serde_json::from_str(config_str)?;

    if let Some(ae) = root
        .get_mut("audio_encoder")
        .and_then(|v| v.as_object_mut())
    {
        if ae.get("num_codebooks").is_none()
            && let Some(val) = ae.remove("n_codebooks")
        {
            ae.insert("num_codebooks".to_string(), val);
        }

        let frame_rate = if ae.get("frame_rate").is_none() {
            if let (Some(sr), Some(ratios)) = (
                ae.get("sampling_rate").and_then(|v| v.as_u64()),
                ae.get("downsampling_ratios").and_then(|v| v.as_array()),
            ) {
                let product: u64 = ratios.iter().filter_map(|r| r.as_u64()).product();
                if product > 0 {
                    let fr = sr / product;
                    ae.insert("frame_rate".to_string(), serde_json::json!(fr));
                    Some(fr)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            ae.get("frame_rate").and_then(|v| v.as_u64())
        };

        if ae.get("latent_dim").is_none()
            && let Some(val) = ae.remove("hidden_size")
        {
            ae.insert("latent_dim".to_string(), val);
        }

        if ae.get("model_bitrate").is_none()
            && let (Some(n_cb), Some(cb_size), Some(fr)) = (
                ae.get("num_codebooks").and_then(|v| v.as_u64()),
                ae.get("codebook_size").and_then(|v| v.as_u64()),
                frame_rate,
            )
        {
            let bits_per_second = n_cb * (cb_size as f64).log2().round() as u64 * fr;
            ae.insert(
                "model_bitrate".to_string(),
                serde_json::json!(bits_per_second / 1000),
            );
        }
    }

    Ok(Cow::Owned(serde_json::to_string(&root)?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Minimal old-format config (parler-large v1 style).
    fn old_format_config() -> String {
        serde_json::json!({
            "audio_encoder": {
                "num_codebooks": 9,
                "frame_rate": 86,
                "latent_dim": 1024,
                "model_bitrate": 6,
                "sampling_rate": 44100,
                "codebook_size": 1024
            },
            "decoder": { "num_codebooks": 9 }
        })
        .to_string()
    }

    /// Minimal new-format config (parler-mini v1.1 style).
    fn new_format_config() -> String {
        serde_json::json!({
            "audio_encoder": {
                "n_codebooks": 9,
                "hidden_size": 1024,
                "sampling_rate": 44100,
                "downsampling_ratios": [2, 4, 8, 8],
                "codebook_size": 1024
            },
            "decoder": { "num_codebooks": 9 }
        })
        .to_string()
    }

    #[test]
    fn old_format_passes_through() {
        let input = old_format_config();
        let result = normalize_config(&input).unwrap();
        // Should borrow (no modification needed).
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn new_format_gets_converted() {
        let input = new_format_config();
        let result = normalize_config(&input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let ae = &parsed["audio_encoder"];

        assert_eq!(ae["num_codebooks"], 9);
        assert_eq!(ae["latent_dim"], 1024);
        assert!(ae.get("frame_rate").is_some());
        assert!(ae.get("model_bitrate").is_some());

        // frame_rate = 44100 / (2*4*8*8) = 44100 / 512 = 86
        assert_eq!(ae["frame_rate"], 86);
    }

    #[test]
    fn new_format_removes_old_keys() {
        let input = new_format_config();
        let result = normalize_config(&input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let ae = &parsed["audio_encoder"];

        // n_codebooks and hidden_size should be removed (replaced).
        assert!(ae.get("n_codebooks").is_none());
        assert!(ae.get("hidden_size").is_none());
    }

    #[test]
    fn preserves_other_fields() {
        let input = new_format_config();
        let result = normalize_config(&input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

        // decoder section should survive.
        assert_eq!(parsed["decoder"]["num_codebooks"], 9);
    }
}
