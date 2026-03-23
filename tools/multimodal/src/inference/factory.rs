use crate::config::ModelPaths;
use crate::inference::InferenceEngine;
use crate::inference::pipelines::qwen3_5::Qwen35Engine;

pub fn create_engine(model_name: String, paths: ModelPaths) -> Box<dyn InferenceEngine> {
    Box::new(Qwen35Engine::new(model_name, paths))
}
