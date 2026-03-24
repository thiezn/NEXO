

- Be able to provide an image and a text prompt and perform the requested text prompt on the image recognition. For example:
    - explain what is on this image.
    - What is this person's hair color?

What are the strongest models to date for my use, that can reasonably be implemented in Rust. I do not want to integrate with an external API like LMStudio or Ollama. The crate should only load the model and perform inference.

I need to focus on optimal performance on this hardware. Think of using cradle or perhaps rust bindings into MLX.

Do deep research on all of the latest models and experiences with people for high token/per/second and good results

- I will go down the route of very specific models to get the max performance for the task with most speed. Just like the LoRA approach, i will spend time collecting my own datasets for fine tuning to improve performance/results for nice use cases.

- calling tools
- chatting
- summarizing text
- speech to text
- text to speech
- image recognition with text prompts

- Be able to generally chat in english language
- Be able to prompt it with text to output strong structured data for tool calling
