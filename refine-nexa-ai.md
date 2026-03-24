Refine the nexa-ai codebase and prepare step by step instructions on how to implement a new AI model.


# Step 1. Add statistics collection for model loading and inference times.

To evaluate performance of the models, it would be useful to have some basic statistics about the model loading and inference times. Add a statistics module that collects this information to the @nexo-ai/src/statistics directory.

For now, statistics will be in-memory only, but it should be designed in a way that it can be easily extended to support persistent storage in the future (for instance, by writing to a file or a sqlite database).

Look up best practices online for measuring model performance. If there are different metrics for different model categories (for instance, token per second for chat models, or images per second for imagine models), make sure to collect the relevant metrics for each category. If this is the case, consider updating the @model_traits.rs design to enforce collection of the relevant metrics for each model category.

# Step 2. Add statistics to CLI

Update the CLI REPL to:
- include a command to show the collected statistics. This will allow users to see the performance of the currently loaded models and compare them.
- During inference, show the token per seconds and total inference time for each prompt. 

Use nice formatting to make it easy to read and understand. indicatif and related modules can be used to show progress bars and timing information during inference.

# Step 3. Create a registry module.

- Make a clear file structure for the registry, with separate files for the coordinator, the registry itself, and the models.
- Move the related models and functions away from the download module into the registry module. The download module should only contain the functions related to the download functionality.
- Move the coordinator/registry.rs content into the registry module.

# Step 4. Simplify

Run the /simplify skill to make sure our codebase is clean. Focus especially on the file, folder and module structure.

# Step 4. Review and update the @introduction.md file.

The @introduction.md file for nexo-ai describes the initial prompt used to build the crate. It highlights key architectural and design decisions. 

- Review the existing nexo-ai crate code
- Compare the code with the description in the @introduction.md file
- Update the @introduction.md file to reflect the current state of the codebase, including any changes made during the refinement process. This will ensure that the documentation accurately represents the crate's architecture and design.
- Reword the introduction to be more concise and clear, while still covering the key points about the coordinator module, the CLI interface, and the model management logic. Change the tone from a 'prompt' to a more formal documentation/architecture style.

# Step 5. Create SKILL.md file to guide the addition of new models.

launch a subagent, using the /skill-creator skill, with the correct context around the architecture described in @introduction.md file, as well as the key files and modules involved in the model management.

The skill should explain the model traits, registry and model categories (chat, tool, image, talk, listen, imagine, multipurpose) and then provide step by step instructions on how to add a new model to the nexo-ai framework. The instructions should cover:

- How to implement the appropriate trait for the model category (for instance, if it's a chat model, it should implement the ChatModel trait)
- Where to place the code for the model (in the @nexo-ai/src/models directory, in the appropriate category folder, and then in a folder named after the model)
- How to add the model to the registry, including how to define the model manifest and how to implement the necessary functions to load the model and call it from the gateway.
- Step by step instructions on how to actually implement the model ready for inference. Any tokenizer, forwarding, encoder, etc required to run inference. Hints on using candle and how to fill in the gaps if candle doesn't support a specific feature required by the model.
- Mention the @hf_downloader.py script can be used to retrieve information from Hugging Face. Retrieve the required information from Hugging Face (using the hf_downloader.py script)
