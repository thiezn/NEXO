# NEXO AI

`nexo-ai` is the inference runtime crate for the NEXO workspace. It provides a single trait-based API for local Candle models and OpenAI-compatible remote backends, plus registry, lifecycle, download, and statistics layers around that API.

The current architecture is built around two explicit ideas:

- model family and execution backend are separate concepts
- model code is organized by family, with backend-specific implementations below each family
