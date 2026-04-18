# Running LLM Inference with Python

At the moment most of the AI world is concentrated on using Python and C/C++ for inference. Rust has the Candle and mlx-rs crates that allow you to build inference but it's a less mature environment.

nexo-ai has implemented several models using rust candle but the performance is lacking on Apple sillicon and the effort to maintain it is high. Therefore we will instead rely on existing packages and offload the inference to Python.

## MLX LM/VLM/Audio

The community seems to be rallying behind the MLX-lm family of repositories resided here: https://github.com/Blaizzy/mlx-vlm

We will leverage this and use nexo-ai to start the server and invoke queries using the OpenAI REST API it provides.


## Quick tests

After installing mlx-vlm you can run the following command to test the VLM capabilities:

```sh
export HF_ENDPOINT=https://hf-mirror.com
mlx_vlm.generate --model mlx-community/gemma-4-E2B-it-4bit --max-tokens 100 --prompt "Describe what you see and hear" --image datasets/images/mk2_pants_down.png --audio datasets/audio/monkeyinmypocket.wav
```
