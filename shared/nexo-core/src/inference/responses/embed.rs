use serde::{Deserialize, Serialize};

use super::super::usage::TokenUsage;

/// A single embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingVector {
    /// The zero-based order of the vector within the response.
    pub index: usize,

    /// The embedding values.
    pub values: Vec<f32>,
}

/// The response returned for an embedding request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbedResponse {
    /// The embedding vectors returned by the runtime.
    pub vectors: Vec<EmbeddingVector>,

    /// Token usage recorded for the embedding operation.
    pub usage: Option<TokenUsage>,
}

/// The response returned for a tokenization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenizationResponse {
    /// The resulting token ids.
    pub tokens: Vec<u32>,
}

/// The response returned for a detokenization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DetokenizationResponse {
    /// The detokenized text.
    pub text: String,
}
