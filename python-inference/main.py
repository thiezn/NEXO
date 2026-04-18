#!/usr/bin/env python3

# Make sure to set hugging face ot the mirror: export HF_ENDPOINT=https://hf-mirror.com

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load multi-modal model
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# Prepare inputs
image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = ""

# Apply chat template
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image), num_audios=len(audio))

# Generate output
output = generate(model, processor, formatted_prompt, image, audio=audio, verbose=False)
print(output)
