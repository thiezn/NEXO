# Running LLM Inference with Python

At the moment most of the AI world is concentrated on using Python and C/C++ for inference. Rust has the Candle and mlx-rs crates that allow you to build inference but it's a less mature environment.

nexo-ai has implemented several models using rust candle but the performance is lacking on Apple sillicon and the effort to maintain it is high. Therefore we will instead rely on existing packages and offload the inference to Python.

At the moment I'm starting the python server of mlx-vlm and mlx-audio and using the OpenAI compatible API to query it from Rust. However, I think perhaps I can get a cleaner way of doing this my using the python libraries in my own module and create bindings with PYo3. 

## MLX LM/VLM/Audio

The community seems to be rallying behind the MLX-lm family of repositories resided here: https://github.com/Blaizzy/mlx-vlm

We will leverage this and use nexo-ai to start the server and invoke queries using the OpenAI REST API it provides.

## Image generation

This isn't supported by MLX-lm yet but we can use the mflux package for that:

## Quick tests

After installing mlx-vlm you can run the following command to test the VLM capabilities:

```sh
export HF_ENDPOINT=https://hf-mirror.com
mlx_vlm.generate --model mlx-community/gemma-4-E2B-it-4bit --max-tokens 100 --prompt "Describe what you see and hear" --image datasets/images/mk2_pants_down.png --audio datasets/audio/monkeyinmypocket.wav
```


## Alternatief

https://mintlify.wiki/OminiX-ai/OminiX-MLX/image/flux-klein

