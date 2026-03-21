# Image Generator

This tool takes in one or more paragraphs of text and generates an image that captures the essence of the text.

The caller can determine the text LLM processor that will generate the prompt for the image generation. It can set the image generation model and a LoRA to use for the generation. The caller can also specify the desired image dimensions and the number of images to generate. 

The generated image(s) are returned as a base64-encoded string?

In the future this will be a lot more generic tool for other image generation tasks like logo's in a certain style, images of LoRA trained models of people I know so they can feed a picture of themselves and get a consistent avatar returned by a custom trained LoRA on them (will use my iphone library to find people and suprise them).

## Architecture

- Caller takes in a JSON struct, or cli commands with the text, LLM processor, image generation model, LoRA, dimensions and number of images to generate.

- We send this to LM Studio running some good enough text model (qwen/3.5) that will translate the paragraph into a prompt for image generation
- We send the generated prompt to the LUX.2 model in LM Studio

LM Studio is running the OpenAI compatible api on http://localhost:1234

Use reqwest library, tokio and serde for the API calls and JSON parsing.
