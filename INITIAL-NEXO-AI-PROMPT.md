# Nexo-AI

Build the nexo-ai crate.

I have provided folder scaffolding for the crate in @nexo-ai/src.

## Abstraction interface for AI models

I want to create traits for different type of AI models. These traits will provide a structured interface into the actual models. This will allow me to introduce support for other models in the future without having to change the interface into the rest of my code.

Create the following traits in @nexo-ai/src/shared/model_traits.rs:

 ChatModel - text to text for chatting and summarization
 ToolModel - for taking text prompts and outputting structured data for tool calling
 ImageModel - for analyzing images with text prompts, and outputting structured or unstructured data
 TalkModel - for text to speech
 ListenModel - for speech to text
 ImagineModel - For generating images from text prompts, this is different from the image recognition with text prompts, which will be image model

We need a way to expose to the user of the library which models are currently supported, and which ones are loaded in memory. This should be handled in the @nexo-ai/src/coordinator/registry.rs file.

## LoRA fine tuning

We will be fine tuning LoRA models for the ImagineModel and the ToolModel. The ChatModel will be a more general model that is not fine tuned, but can be used for general chatting and summarization. The ToolModel will be fine tuned to output structured data for tool calling, and the ImagineModel will be fine tuned to generate images from text prompts. Think about a good way to allow us to provide the LoRA model we want to use when calling ToolModel and ImagineModel. 

A generic LoRA trait could help decouple this from actual models. I want to be able to define clear categories of LoRA types like 'Hero Image LoRA', 'Background Image LoRA', 'Object LoRA' for the ImagineModel, and 'Tool Calling LoRA' for the ToolModel. This way we can easily swap out different LoRA models for different use cases without having to change the interface into the rest of the code.

Put the trait definition in @nexo-ai/src/shared/lora_traits.rs

## In-Memory Model Management - Coordinator

We will be running nexo-ai on a local macbook M1/M4, so we need to be very mindful of the memory constraints. We will implement a model management system that allows us to load and unload models from memory on demand. This will allow us to have multiple models available for different use cases without having to have them all loaded in memory at the same time.

We should use the nexo-ai.toml config file to:
    - define the default model per model category.
    - define which model categories should be loaded by default on startup.

The coordinator module will be providing an interface to load/unload models from memory, and to get the currently loaded model for each category. The coordinator will also handle the logic for switching between different LoRA models for the ToolModel and ImagineModel. It will 

## CLI interface

The nexa-ai CLI interface is behind a feature flag.

the code for starting the cli is in @nexo-ai/src/cli/base.rs.

It will have the following commands:

```bash
nexo-ai pull
nexo-ai list
nexo-ai start
```

### nexo-ai pull

This will generate the .nexo/nexo-ai.toml configuration file with the default configuration for the models. It will look at the .nexo/local_models directory to see if the supported models are already downloaded, if not it will download them.

There is a --force flag to force re-downloading the given model or models.

```bash
nexo-ai pull --force <model-name or model-type>
```

- model-type can be chat, tool, image, talk, listen, imagine or all.
- model-name can be the specific name of the model as supported in the tool.

### nexo-ai list

This will list the currently supported models, downloaded models and their configuration. It will look at the .nexo/nexo-ai.toml configuration file to see which models are configured, and then check the .nexo/local_models directory to see if they are downloaded. It will output a table with the model name, type, and whether it is downloaded or not.

### nexo-ai start

This will initialize the default loaded models (chat and tool) and then provide an interactive REPL for issuing commands to the models.

/start categories chat,tool
/config default-chat "model-name"
/list models


The code for the REPL is in @nexo-ai/src/cli/repl.rs. It will use the coordinator to get the currently loaded models and call the appropriate methods on them based on the user input. The REPL commands code should go into separate files per command in the @nexo-ai/src/cli/commands directory.

### Model invocation commands

These commands will invoke the currently running models. The cli will print an error message if there is no model loaded for the given command.

/chat What is the weather like?
/tool generate_image --prompt "a cute cat"
/tool switch_model image
/tool generate_image --prompt "a cute cat"
/tool switch_model chat
/chat What is the weather like?
/talk tell me a story
/listen "audio.wav"

### Crates to use

- Candle-core and candle-nn for model loading and inference. Only with metal feature enabled, no cuda.
- The @shared/utl-helpers crate for common CLI utilities and helpers. Use the /cli-tool-builder skill.
- viuer for displaying images in the terminal
- rodio for audio playback in the terminal (wraps around cpal and symphonia for audio decoding)
- cpal for audio recording in the terminal.
- symphonia and hound for audio processing in the terminal (for instance to convert input audio files to the correct format for the listen model, and to convert output audio from the talk model to a playable format)
- hf-hub for downloading models from Hugging Face and managing the local model cache.

## Nexo Gateway integration

The Nexo Gateway will be the main interface into the Nexo-AI crate. The gateway will use the traits defined in the Nexo-AI crate to call the appropriate models for different use cases. NEXO gateway will import the crate without the cli feature.

## Models layout

We will be adding and removing support for different models over time, so we need to have a clear layout for where the code for each model will go. We should have a separate module for each model in the @nexo-ai/src/models directory. Each module should implement the appropriate trait for the model category it belongs to. For instance, the chat models should implement the ChatModel trait, the tool models should implement the ToolModel trait, and so on.

The @nexo-ai/src/models directory has a folder per model category, and then inside each folder there is a folder per model. For instance, @nexo-ai/src/models/chat/image/google-gemma3 is the folder for the Google Gemma 3 chat model.

In the future we might want to use a model that can be used for multiple categories, for instance a model that can be used for both chatting and tool calling. In that case we can have the model implement multiple traits and then we can decide which trait to use when calling the model from the gateway. These models should go in the multipurpose folder, for instance @nexo-ai/src/models/multipurpose/model-name.

## Final notes

- Add unit tests and run coverage report `cargo llvm-cov --no-cfg-coverage --skip-functions --package nexo-ai`
- Run /simplify after everything is fully working to clean up the code and remove any unnecessary complexity.
- We already have similar code in the @tools/ai folder, but I want a clean slate for the nexo-ai crate to avoid any technical debt from the old code. We can of course reuse some of the code and logic from the old code, but we should not copy and paste it directly into the new crate. We should take the time to refactor and improve the code as we go along.
