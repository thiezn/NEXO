use serde::{Deserialize, Serialize};

/// Controls whether a response should be buffered or streamed.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum StreamingMode {
    /// Buffer the full response and emit it only when complete.
    #[default]
    Buffered,

    /// Emit incremental streamed chunks as they become available.
    Streaming,
}

/// A structured output constraint applied during generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum OutputConstraint {
    /// No output constraint is applied.
    None,

    /// Constrain output to a JSON Schema document.
    JsonSchema(serde_json::Value),

    /// Constrain output to a regular expression.
    Regex(String),

    /// Constrain output to a Lark grammar.
    LarkGrammar(String),
}

/// Sampling and decoding controls applied to a generation request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SamplingConfig {
    /// The maximum number of output tokens to generate.
    pub max_output_tokens: Option<usize>,

    /// The temperature used during token sampling.
    pub temperature: Option<f32>,

    /// The top-p nucleus sampling threshold.
    pub top_p: Option<f32>,

    /// The top-k candidate cutoff.
    pub top_k: Option<u32>,

    /// The minimum probability threshold applied during sampling.
    pub min_p: Option<f32>,

    /// The frequency penalty applied to repeated tokens.
    pub frequency_penalty: Option<f32>,

    /// The presence penalty applied to previously seen tokens.
    pub presence_penalty: Option<f32>,

    /// The repetition penalty applied during decoding.
    pub repetition_penalty: Option<f32>,

    /// The deterministic random seed, if one is requested.
    pub seed: Option<u64>,

    /// Explicit stop sequences that terminate decoding when matched.
    pub stop_sequences: Vec<String>,
}
