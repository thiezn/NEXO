//! Flux.2 model variant configuration.

use super::transformer::Flux2Config;
use super::vae::Flux2VaeConfig;
use anyhow::Result;
use std::path::PathBuf;

/// Identifies which Flux.2 variant we are running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxVariant {
    Klein4B,
    Klein9B,
    Dev,
}

impl FluxVariant {
    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            "flux-2-klein-4b" => Some(Self::Klein4B),
            "flux-2-klein-9b" => Some(Self::Klein9B),
            "flux-2-dev" => Some(Self::Dev),
            _ => None,
        }
    }

    /// Return the transformer config if known, or infer from safetensor headers.
    pub fn transformer_config(
        &self,
        transformer_paths: &[PathBuf],
    ) -> Result<Flux2Config> {
        match self {
            Self::Klein4B => Ok(Flux2Config::klein()),
            _ => infer_transformer_config(transformer_paths),
        }
    }

    /// Return the VAE config. Klein models share the same VAE.
    pub fn vae_config(&self) -> Result<Flux2VaeConfig> {
        match self {
            Self::Klein4B | Self::Klein9B => Ok(Flux2VaeConfig::klein()),
            // Dev VAE uses the same architecture as Klein; the larger file size
            // (321 MB vs 160 MB) is due to F32 weight precision, not a different config.
            Self::Dev => Ok(Flux2VaeConfig::klein()),
        }
    }

    /// Whether this variant uses guidance embedding (non-distilled).
    pub fn has_guidance_embed(&self) -> bool {
        match self {
            Self::Klein4B | Self::Klein9B => false,
            Self::Dev => true,
        }
    }

    /// Default number of denoising steps for this variant.
    pub fn default_steps(&self) -> u32 {
        match self {
            Self::Klein4B | Self::Klein9B => 4,
            Self::Dev => 28,
        }
    }

    /// Default guidance scale for this variant.
    pub fn default_guidance(&self) -> f64 {
        match self {
            Self::Klein4B | Self::Klein9B => 0.0,
            Self::Dev => 3.5,
        }
    }
}

/// Infer transformer config by reading safetensor headers.
///
/// Reads the JSON header from the first safetensor file to discover:
/// - `depth` from `transformer_blocks.N.` keys
/// - `depth_single_blocks` from `single_transformer_blocks.N.` keys
/// - `hidden_size` from `x_embedder.weight` shape
/// - `num_heads` from head dim via `norm_q.weight`
/// - `guidance_embed` from presence of `guidance_embedder` keys
fn infer_transformer_config(paths: &[PathBuf]) -> Result<Flux2Config> {
    if paths.is_empty() {
        anyhow::bail!("no transformer files provided for config inference");
    }

    // Read safetensor header from the first file
    let file = std::fs::File::open(&paths[0])?;
    let mut reader = std::io::BufReader::new(file);

    // Safetensors format: first 8 bytes = header length (little-endian u64)
    use std::io::Read;
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Cap at 100 MB to avoid OOM on corrupted files
    if header_len > 100_000_000 {
        anyhow::bail!("safetensor header too large: {header_len} bytes");
    }

    let mut header_buf = vec![0u8; header_len];
    reader.read_exact(&mut header_buf)?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

    let keys: Vec<&str> = header
        .as_object()
        .map(|o| o.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();

    // Count double blocks
    let depth = keys
        .iter()
        .filter_map(|k| {
            k.strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(5);

    // Count single blocks
    let depth_single = keys
        .iter()
        .filter_map(|k| {
            k.strip_prefix("single_transformer_blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(20);

    // Infer hidden_size from x_embedder.weight shape [hidden_size, in_channels]
    let hidden_size = header
        .get("x_embedder.weight")
        .and_then(|v| v.get("shape"))
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(3072);

    // Infer in_channels from x_embedder.weight shape [hidden_size, in_channels]
    let in_channels = header
        .get("x_embedder.weight")
        .and_then(|v| v.get("shape"))
        .and_then(|s| s.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(128);

    // Infer num_heads from norm_q.weight shape [head_dim]
    let head_dim = keys
        .iter()
        .find(|k| k.contains("norm_q.weight"))
        .and_then(|k| header.get(*k))
        .and_then(|v| v.get("shape"))
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(128);
    let num_heads = if head_dim > 0 {
        hidden_size / head_dim
    } else {
        24
    };

    // Infer context_in_dim from context_embedder.weight shape [hidden_size, context_in_dim]
    let context_in_dim = header
        .get("context_embedder.weight")
        .and_then(|v| v.get("shape"))
        .and_then(|s| s.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(7680);

    // Check for guidance embedding
    let guidance_embed = keys
        .iter()
        .any(|k| k.contains("guidance_embedder"));

    tracing::info!(
        depth,
        depth_single,
        hidden_size,
        num_heads,
        in_channels,
        context_in_dim,
        guidance_embed,
        "inferred Flux.2 transformer config from safetensor headers"
    );

    Ok(Flux2Config {
        in_channels,
        vec_in_dim: 0,
        context_in_dim,
        hidden_size,
        mlp_ratio: 3.0,
        num_heads,
        depth,
        depth_single_blocks: depth_single,
        axes_dim: vec![32, 32, 32, 32],
        theta: 2000,
        guidance_embed,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn variant_from_model_name() {
        assert_eq!(
            FluxVariant::from_model_name("flux-2-klein-4b"),
            Some(FluxVariant::Klein4B)
        );
        assert_eq!(
            FluxVariant::from_model_name("flux-2-klein-9b"),
            Some(FluxVariant::Klein9B)
        );
        assert_eq!(
            FluxVariant::from_model_name("flux-2-dev"),
            Some(FluxVariant::Dev)
        );
        assert_eq!(FluxVariant::from_model_name("unknown"), None);
    }

    #[test]
    fn klein_4b_has_known_config() {
        let cfg = FluxVariant::Klein4B.transformer_config(&[]).unwrap();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.depth, 5);
        assert_eq!(cfg.depth_single_blocks, 20);
        assert!(!cfg.guidance_embed);
    }

    #[test]
    fn klein_default_steps() {
        assert_eq!(FluxVariant::Klein4B.default_steps(), 4);
        assert_eq!(FluxVariant::Klein9B.default_steps(), 4);
        assert_eq!(FluxVariant::Dev.default_steps(), 28);
    }

    #[test]
    fn guidance_embed_variants() {
        assert!(!FluxVariant::Klein4B.has_guidance_embed());
        assert!(!FluxVariant::Klein9B.has_guidance_embed());
        assert!(FluxVariant::Dev.has_guidance_embed());
    }

    #[test]
    fn default_guidance_values() {
        assert_eq!(FluxVariant::Klein4B.default_guidance(), 0.0);
        assert_eq!(FluxVariant::Dev.default_guidance(), 3.5);
    }

    #[test]
    fn vae_config_for_all_variants() {
        for variant in [FluxVariant::Klein4B, FluxVariant::Klein9B, FluxVariant::Dev] {
            let cfg = variant.vae_config().unwrap();
            assert_eq!(cfg.latent_channels, 32);
        }
    }
}
