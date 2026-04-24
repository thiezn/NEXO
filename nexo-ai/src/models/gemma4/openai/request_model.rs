use std::path::Path;

/// Resolve the default model identifier sent to `mlx_vlm.server`.
///
/// By default nexo-ai points MLX VLM at the local nexo-managed model directory.
/// Callers can still override this explicitly when they need to target a repo id.
pub fn default_request_model_id(_model_name: &str, model_dir: &Path) -> String {
    model_dir.to_string_lossy().to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn e2b_defaults_to_local_path() {
        let model_id = default_request_model_id(
            "mlx-gemma-4-e2b-it-8bit",
            Path::new("/tmp/mlx-gemma-4-e2b-it-8bit"),
        );

        assert_eq!(model_id, "/tmp/mlx-gemma-4-e2b-it-8bit");
    }

    #[test]
    fn other_models_fall_back_to_local_path() {
        let model_id = default_request_model_id("mlx-gemma-4-31b-it-4bit", Path::new("/tmp/model"));

        assert_eq!(model_id, "/tmp/model");
    }
}
