

Refactor "Flux.2 model loading error" session stopped as credits were exhaused.

It's a big mess still.

I need to instruct it to get rid of runtime_preference completely. I don't want to support different runtimes for the same model, i want opinionated choice for myself on what runtime to use. TRuntime will be defined in the nexo-model-mgmt.


Also, wtf is the preferred_capabilities. I need to kill this completely as well. it doesn't make any sense? Probably introduced for the agent loop router?


RuntimeImplementation and ModelRuntimeImplementation also seems to be the same thing, wtf


ModelDataType and ManifestModelDataType are the same thing, move to core? or at least import it from nexo-model-mgmt?


Kill nexo-ai.toml. nexo-ai should not have a cli with toml files. This should only live in nexo-node, not in the library crate.



The runtime_config_from_manifest_binding function seems to expose how shitty the the three runtime config loading system is. All three seems to have a different way of handling things. you've got AnyTtsManifestEngine, MoldManifestLoader and MistralRsModelConfig that almost look like the same thing. Also exposes ModelRuntimeImplementation and RuntimeImplementation, do they really have to be separate?



Since my credits are exhausted and AI seemingly is steering me into bigger messes often, gather your confidence back and solve this yourself! AI is nice, but I need to understand this and get a clean codebase. I CAN do this, this is not any difficult math type of thing, its just some glue code. I am good enough for this stuff.


Read this article to ground my knowledge again for rust: https://medium.com/@carlmkadie/nine-ways-to-do-inheritance-in-rust-a-language-without-inheritance-14825bf1e215



## Decision for now


nexo-model-mgmt starts to become intertwined with nexo-ai. The main reason i wanted this separate crate is so I could have different binaries download models. So gateway, client and node. This is nice but actually not a good reason enough to cause all kinds of abstractions. Especially with all the runtimes it starts to become more tighly coupled and eventually, the nexo-gateway, node or client should not have an opinion on the runtine. Only nexo-ai should be allowed to have this.

So the manifest and runtime belong in nexo ai. Download is a bit more fuzzy but i think i want to opt for download being a library only, and then nexo-node will only get a cli interface into the download. nexo-node will also hold the UX for model manifest management.
