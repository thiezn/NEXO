

Current State
nexo-ai.toml (~/.nexo/nexo-ai.toml):

active_models — category→model map (persisted)
startup_categories — which categories to auto-load (default: ["chat", "talk"])
models — per-model settings (temperature, max_tokens, etc.)
node.toml (~/.nexo/node.toml):

Gateway connectivity (url, node_id, auth, etc.)
available_models — populated dynamically at startup but stored in the toml
Problems: nexo-node loads AiConfig directly, meaning nexo-ai.toml settings leak into node usage. available_models is redundant in config. No default model auto-selection. Node doesn't advertise per-model capabilities to gateway. Node doesn't auto-load any models at boot.

Questions
Q1: Config file structure — Should nexo-node's model config live inside node.toml itself, or a separate file like ~/.nexo/nexo-node.toml? For example:


# Option A: model section inside node.toml
[gateway]
url = "ws://127.0.0.1:6969"
node_id = "my-node"

[models]
# default model preferences and per-model settings here

# Option B: keep node.toml for connectivity, separate file for models
Q2: Shared model settings — Per-model settings (temperature, max_tokens, max_context_tokens, etc.) — should these be shared between nexo-ai CLI and nexo-node? Or should each tool have independent model settings? If you tweak temperature for gemma-4-e4b-it in one, should the other see it?

Q3: Default model preferences — You mentioned a [default_models] section. Should this be per-category, or per-model-family? For example:


# Option A: per-category
[default_models]
chat = "gemma-4-e4b-it"
imagine = "z-image-turbo"




# Option B: per-family (auto-selects best variant)
[default_models]
chat = "gemma4"  # picks best available gemma4 variant
imagine = "z_image"  # picks best available z_image variant
Q4: Auto-start scope — You said nexo-node should start defaults for chat, tools, and image. Currently the Coordinator loads one model per category. Since gemma4 supports chat+tool+image, loading one gemma4 model covers all three. Should the node try to load one model per category, or be smart about multi-category models (load one gemma4 = done for chat/tool/image)?

Q5: Gateway capability advertisement — Currently the node only tells the gateway coarse capabilities ("llm", "vision"). Should it instead send a full manifest per model? Something like:


{
  "available_models": [
    {"name": "gemma-4-e4b-it", "categories": ["chat", "tool", "image"], "size_gb": 14.9, "loaded": true},
    {"name": "z-image-turbo", "categories": ["imagine"], "size_gb": 11.9, "loaded": false}
  ]
}
Q6: Gateway model management — Should the gateway be able to tell a node "load model X" even if the node didn't choose it as a default? And conversely, should the node refuse if it doesn't have the model downloaded? (This already works via model.load / model.unload protocol, but I want to confirm the intent.)

Q7: Startup categories — You want nexo-node to auto-start chat+tool+image. Should nexo-ai start (the standalone CLI) keep its own separate startup_categories, or should both tools share the same "start these by default" logic?




Q1: Option A - It should reside in node.toml itself.
Q2: These should not be shared. nexo-ai config toml should only be loaded when using nexo-ai cli. The whole config file should be ignored when running nexo-node
Q3: should be per category, not per family
Q4: it should be smart about multi category models, only load one
Q5: we should remove the coarse capabilities completely. The gateway should be fully aware of the exact available categories and make a decision what it needs
Q6: Yes the gateway should be in full control of load/unload of models on nexo-node instances
Q7: Ye, nexo-ai start the standalone cli need it's own startup_categories in nexo-ai.toml. This should be completely independent from nexo-node

Go ahead and implement it all.
