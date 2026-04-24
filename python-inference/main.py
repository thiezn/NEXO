#!/usr/bin/env python3

from pathlib import Path

import mlx_vlm
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model
from mlx_vlm.prompt_utils import apply_chat_template
from rich.console import Console
from rich.markdown import Markdown

console = Console()


# Make sure to set hugging face ot the mirror: export HF_ENDPOINT=https://hf-mirror.com

DATASET_FOLDER = "/Users/Mathijs.Mortimer/Development/nexo/datasets"
MODELS_FOLDER = "/Users/Mathijs.Mortimer/.nexo/local_models"
WHISPER_MODEL = "mlx-whisper-large-v3-turbo-asr-fp16"
GEMMA_MODEL = f"{MODELS_FOLDER}/mlx-gemma-4-e2b-it-8bit"
GEMMA_LARGE_MODEL = f"{MODELS_FOLDER}/mlx-gemma-4-31b-it-4bit"  # This does not have audio.
VOXTRAL_MODEL = f"{MODELS_FOLDER}/mlx-voxtral-4b-tts-2603-bf16"


def tts_model():
    # Load TTS model

    print(f"Loading TTS model {VOXTRAL_MODEL}...")
    model = load_model(Path(VOXTRAL_MODEL))

    # Generate speech
    for result in model.generate("I don't know about you, but i am going to bed", verbose=True):
        print(f"Generated {result.audio.shape[0]} samples")
        # result.audio contains the waveform as mx.array
        # https://github.com/Blaizzy/mlx-audio/blob/57555a0e48da6e24faac8aa3db17ab7e41354e3f/examples/omnivoice_clone_demo.py
        audio = np.array(result.audio)
        sf.write("output.wav", audio, result.sample_rate)


def vlm_model():
    """Multi-modal model

    The smaller gemma model can analyze audio, the larger model doesnt but is a lot better.
    """
    # Load multi-modal model
    print(f"Loading model from {GEMMA_LARGE_MODEL}...")
    model, processor = mlx_vlm.load(GEMMA_LARGE_MODEL)
    # model, processor = mlx_vlm.load(GEMMA_MODEL)
    print("Model loaded successfully.")
    config = model.config

    # Prepare inputs
    image = [f"{DATASET_FOLDER}/images/mk2_pants_down.png"]
    audio = [f"{DATASET_FOLDER}/audio/monkeyinmypocket.wav"]
    prompt = "What is in the image and audio?"

    # Apply chat template
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image), num_audios=len(audio))

    # Generate output
    output = mlx_vlm.generate(model, processor, formatted_prompt, image, audio=audio, verbose=False, max_tokens=4096)  # type: ignore
    print("Output generated successfully.")
    markdown = Markdown(output.text)
    console.print(markdown)


def image_generation():
    """Image generation"""
    from mflux.models.flux2 import Flux2Initializer, Flux2Klein
    from mflux.models.z_image import ZImageTurbo

    model_path = "/Users/Mathijs.Mortimer/.nexo/local_models/flux-2-klein-4b"
    model = Flux2Klein(model_path=model_path)
    # model_path = "/Users/Mathijs.Mortimer/.nexo/local_models/z-image-turbo"
    # model = ZImageTurbo(model_path=model_path)

    image = model.generate_image(
        prompt="A puffin standing on a cliff",
        seed=42,
        num_inference_steps=9,
        width=1280,
        height=500,
    )
    image.save("puffin.png")


if __name__ == "__main__":
    # tts_model()
    # vlm_model()
    image_generation()
