//! Flux.2 model configuration and variant detection.
//!
//! Supports:
//! - Klein-4B (Apache 2.0, distilled, `hidden_size=3072`, 5+20 blocks)
//! - Klein-9B (Non-Commercial, distilled, `hidden_size=4096`, 8+24 blocks)

/// Which Flux.2 variant this model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxVariant {
    /// Klein-4B: smaller, Apache-licensed.
    Klein4B,
    /// Klein-9B: larger, non-commercial.
    Klein9B,
}

impl FluxVariant {
    /// Detect the variant from the model name.
    ///
    /// Looks for `"9b"` (case-insensitive) to select Klein-9B;
    /// everything else defaults to Klein-4B.
    pub fn from_name(name: &str) -> Self {
        if name.to_ascii_lowercase().contains("9b") {
            Self::Klein9B
        } else {
            Self::Klein4B
        }
    }

    /// Return the transformer configuration for this variant.
    pub fn transformer_config(&self) -> Flux2Config {
        match self {
            Self::Klein4B => Flux2Config::klein(),
            Self::Klein9B => Flux2Config::klein_9b(),
        }
    }

    /// Return the VAE configuration (same for all variants).
    pub fn vae_config(&self) -> Flux2VaeConfig {
        Flux2VaeConfig::klein()
    }
}

// ---------------------------------------------------------------------------
// Transformer config
// ---------------------------------------------------------------------------

/// Flux.2 transformer configuration.
#[derive(Debug, Clone)]
pub struct Flux2Config {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub guidance_embed: bool,
}

impl Flux2Config {
    /// Configuration for Flux.2 Klein-4B (Apache 2.0, distilled).
    pub fn klein() -> Self {
        Self {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 7680,
            hidden_size: 3072,
            mlp_ratio: 3.0,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 20,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: false,
        }
    }

    /// Configuration for Flux.2 Klein-9B (Non-Commercial, distilled).
    /// Larger Qwen3 encoder (`hidden_size=4096`, `joint_attention_dim=12288`).
    pub fn klein_9b() -> Self {
        Self {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 12288, // 4096 * 3 (Qwen3 hidden_size stacked 3x)
            hidden_size: 4096,
            mlp_ratio: 3.0,
            num_heads: 32,
            depth: 8,
            depth_single_blocks: 24,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// VAE config
// ---------------------------------------------------------------------------

/// Flux.2 VAE configuration.
#[derive(Debug, Clone)]
pub struct Flux2VaeConfig {
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub use_post_quant_conv: bool,
    /// Number of patchified channels: `latent_channels * patch_h * patch_w`.
    pub patchified_channels: usize,
    pub batch_norm_eps: f64,
}

impl Flux2VaeConfig {
    pub fn klein() -> Self {
        Self {
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 32,
            norm_num_groups: 32,
            use_post_quant_conv: true,
            patchified_channels: 32 * 2 * 2, // 128
            batch_norm_eps: 0.0001,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn variant_from_name_klein_4b() {
        assert_eq!(FluxVariant::from_name("flux2-klein"), FluxVariant::Klein4B);
        assert_eq!(
            FluxVariant::from_name("flux2-klein-4b"),
            FluxVariant::Klein4B
        );
        assert_eq!(
            FluxVariant::from_name("flux2-klein:q8"),
            FluxVariant::Klein4B
        );
    }

    #[test]
    fn variant_from_name_klein_9b() {
        assert_eq!(
            FluxVariant::from_name("flux2-klein-9b"),
            FluxVariant::Klein9B
        );
        assert_eq!(
            FluxVariant::from_name("flux2-klein-9B"),
            FluxVariant::Klein9B
        );
        assert_eq!(FluxVariant::from_name("flux2-9b:q8"), FluxVariant::Klein9B);
    }

    #[test]
    fn klein_4b_config_dimensions() {
        let cfg = Flux2Config::klein();
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.hidden_size / cfg.num_heads, 128); // head_dim
        assert_eq!(cfg.depth, 5);
        assert_eq!(cfg.depth_single_blocks, 20);
        assert_eq!(cfg.axes_dim, vec![32, 32, 32, 32]);
        assert_eq!(cfg.theta, 2000);
        assert!(!cfg.guidance_embed);
    }

    #[test]
    fn klein_4b_context_dim_matches_qwen3() {
        let cfg = Flux2Config::klein();
        assert_eq!(cfg.context_in_dim, 7680);
        assert_eq!(cfg.context_in_dim, 2560 * 3);
    }

    #[test]
    fn klein_9b_config_dimensions() {
        let cfg = Flux2Config::klein_9b();
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.hidden_size / cfg.num_heads, 128);
        assert_eq!(cfg.depth, 8);
        assert_eq!(cfg.depth_single_blocks, 24);
        assert_eq!(cfg.context_in_dim, 12288);
        assert_eq!(cfg.context_in_dim, 4096 * 3);
        assert!(!cfg.guidance_embed);
    }

    #[test]
    fn klein_4b_mlp_sizes() {
        let cfg = Flux2Config::klein();
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        assert_eq!(mlp_sz, 9216);
        assert_eq!(h_sz * 3 + mlp_sz * 2, 27648); // single fused projection
        assert_eq!(h_sz + mlp_sz, 12288); // single output projection
    }

    #[test]
    fn klein_4b_vec_in_dim_zero() {
        let cfg = Flux2Config::klein();
        assert_eq!(cfg.vec_in_dim, 0);
    }

    #[test]
    fn vae_config() {
        let cfg = Flux2VaeConfig::klein();
        assert_eq!(cfg.latent_channels, 32);
        assert_eq!(cfg.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(cfg.patchified_channels, 128);
        assert!(cfg.use_post_quant_conv);
    }

    #[test]
    fn variant_config_roundtrip() {
        let v = FluxVariant::Klein4B;
        let cfg = v.transformer_config();
        assert_eq!(cfg.hidden_size, 3072);

        let v = FluxVariant::Klein9B;
        let cfg = v.transformer_config();
        assert_eq!(cfg.hidden_size, 4096);
    }
}
