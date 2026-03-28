# Mijn nieuwe actielijst

Even afgeleid van MRPF (wilde alleen maar heeeel even een logootje genereren, vervolgens zit ik openclaw na te bouwen, image generation pipelines te bouwen, scummvm assets extraction, vision OS game aan het maken in rust en swift en epub image generator te fixen...).

- Add sqlite storage for kv cache on the nexo-nodes. The nexo-gateway zal de prefill doen, maar als prefix veranderd moeten we een nieuwe kv cache opbouwen. we moeten id's genereren voor de verschillende prefill sets die we hebben, vervolgens kan de gateway de node zeggen welke prefill hij moet gebruiken. Als hij in zn sqlite de prefill id vind gebruikt hij de kv cache, als hij m niet vind vraagt ie aan de gateway om een nieuwe volle chat voor prefill te sturen. Dit moet elke keer gebeuren als er een andere sessie word gestart dan toevallig al in geheugen zou kunnen zitten.
- ok weer een nieuw model, is t beter dan parler? https://mistral.ai/news/voxtral-tts Nog geen gguf dus ff wachten
- investigate if we can re-implement and train https://github.com/pushkarjajoria/Text-Conditioned-Drumbeat-Generation?tab=readme-ov-file ourselves? They havent released the weights, not sure how compute intensive it would be to train that? Might be modest, and i can definitely scrape midi drums to train.
- Ik vermoed dat Qwen3-TTS het makkelijkst is. Er is al iemand die het aan de praat heeft in candle: https://github.com/TrevorS/qwen3-tts-rs en hier eentje die gguf gebruikt met een andere framework. ik wil candle EN GGUF. Hier iemand die mlx-rs gebruikt in plaats van candle. https://github.com/second-state/qwen3_tts_rs 't blijft aantrekkelijk om mlx maar de bindings zijn dus nog experimenteel. 
- Z-Image wellicht moet ik switchen naar GGUFF met quant want we zitten nu op F32 wat dubbel zoveel memory zal vragen? z-image-turbo-q5_k_m.gguf
-  rundown van verschillende modellen: https://medium.com/diffusion-doodles/model-rundown-z-image-turbo-qwen-image-2512-edit-2511-flux-2-dev-fc787f5e87ad
- Investigate Z-Image-turbo and especially its editing version: https://medium.com/@302.AI/z-image-turbo-vs-flux-2-dev-heres-what-we-found-e7a31327be40. Theres a whole site with z-image stuff https://z-image.me/en/resources/pixel-art-lora
- Can i print out text tokens as they come in, seems to print it now at the end.
- Implement KV Cache management. Probably easiest is to 'guess' the cache by incoming tokens, and clear them when it hits a certain threshold. I can also on a cron job clear stale sessions, or summarize sessions, store the summary in a session history and then clear it. Then as a client we can perhaps pull in old summaries to provide an initial context for the model. Check the KV_CACHE_MANAGEMENT.md for more details. It seems the prefill bit is where i can load things? It mentions this flow:  summary → clear/recreate → compact prefill. The best strategy is probably this approach, managing this outside the models in my rust code by counting incoming tokens:

    TTL = Time To Live
    Compaction = summarize/compress older context, then rebuild a fresh shorter context from that summary
    LRU = Least Recently Used


I've done work on it, this is built in phase 1:
    KvCacheState trait (shared/templates/kv_cache.rs)
    pub trait KvCacheState {
        fn cache_token_count(&self) -> usize;
        fn clear_cache(&mut self);
        fn truncate_to(&mut self, token_count: usize);
    }
    Phase 1 (this work): Define trait. Models implement with stub (clear only). Future: real prefix caching.


- Have to check if the memory_estimate_bytes are correct. I think this is just a guess of claude? We should run it, push through a few prompt to get the kv cahche to build up and see how much mem it consumes.
- Ask Grok or OpenAI to list all the things we can tweak in a model inference. Things like
- paginated reading view is taking some compute. Make this hydrated so we can store already computed pages in SwiftData. It will need to recompute lokely when we change the font size.
- Fix switching between pagination and continuous reading that it resets. need to anchor it in some way to the ground truth of the book, perhaps the specific paragraph or even sentence (since pagination splits it into multiple paragraphs?)
- Build interface with bluetooth naar Rowan's piano zodat hij kan zien wat voor midi hij speelt, soort guitar hero iets?
- Update the Swift UX claude skill to use progressive disclosre for the different components
- Review my epub exporter. It now included file paths AND base64 data. I want to change this to only generate one or the other.
- Hook in my raspberry pi with audio hat and make it a full flegded client to my gateway. Then it can launch tools, and the gateway can send it processed drum loops etc.
- Add shared helper crate for audio processing using Symphonia. We can use this in my speech to text/text to speech, text to beat generation tools. It could also become a separate tool that can be invoked by my gateway for changing audio files.
- Add helper crate of image transformations, similar to my audio transformations.
- Build shared lbirary for my websocket types. put them in a folder called schema, and make a folder hierarchy that resembles the nested json structure. Read up on the two different mod.rs styles of rust to see if the new one makes more organizational sense.
- Implement Screen Time tokens
- Implement note taking and cleanup tool
- Implement cron tool
- Implement shortcut so I can talk to it.
- Implement companion mode where I can talk to it whilst on a run and create notes.
- Convert the scumm costume format into format supported by, Swift Spritekit, Swift RealityKit Planes
- Use AutoResearch from Karpathy to improve a bunch o stuff.
- Add NES, Delores/thimbleweed and SNES extraction?
- Think of a way to use my local model tools. I think at the moment i will load a model in memory, run inference once, and then close the model. I need a way to start and stop them, eventually controlled by the gateway. I could think of different setup, i might have text to text model running always, but then if i need alot of memory for some image generation, i stop the text-to-text, start the other, run my batch, shut dfown and then restart the text-to-text.
- It tried to get huggingface manifest, but couldn't as he didn't have a token when building img-to-text. This meant it guessed it himself so that might be a common problem with a bunch more of my models. need a prompt to retrieve and validate all the manifests. UPDATE Stored hugging face token locally and instructed it to use curl which worked for the multimodal build.
- Create cli and lib with Rust Image and viuer to mutate pixel art like pallet changing, resizing, cropping, cleaning up, etc. This can be used for my game extraction tools, but also as a standalone tool for mutating images for other purposes. At minimum I want the nearest upscaling algorithm. https://johanneskopf.de/publications/pixelart/supplementary/multi_comparison.html and this one 2xSaI interpolation algorithm, and this one RotSprite for rotation (https://github.com/tversteeg/rotsprite RUST!) or extrac it from chuot . There might be other cool things in chout we could use. It also has nearest neighbor. Do i need rust-gpu to execute shaders? Are shaders used for processing images? here's rotation: https://codeberg.org/fosk/chuot/src/branch/main/shaders/rotation.wgsl and neirest neighbor: https://codeberg.org/fosk/chuot/src/branch/main/shaders/nearest_neighbor.wgsl use rust wgpu probably for this. There must be tons of shaders i can quikcly test and adopt for my use case? for instance wgpu has built in: FilterMode::Nearest. https://rust-gpu.github.io
- WebSocket interface so I can get my LoRA training pipeline working with review feedback.
- Make Question cards showing an image, tap an answer and horizontally scroll through them and submit. Backend stores my results
- Build some kind of storage system for NEXO. Use sqlite, and rust sqlx to do type safety. Have a production sqlite database and a dev database. PRoduction NEXO instance should run on my macbook M1, dev instance is on my M4. I'd like to be able to iterate quickly and in parallel so i might have to consider creating separate sql databases per feature. This could then avoid complicating types and structs into one big entangled thing?
- gguf with lama apparently performs a lot better than candle on ML. Consider rebuilding using that. For now, lets first get my models doing some actual work and integrated into NEXO
- for the local inference cli's, can we have a nice tokens/sec counter?


##  Run auto research using my local chat model.

- Read in literature like hackers manifesto
- can we push it in and get it cached, doing something with prefill?
- run 100 image generations with different prompts.
- send all images for human review
- randomize the image runs between auto research so iOS app doesn’t know which run it was
- in the morning review them and feed back into the loop
- the loop scores certain datasets and image tags and incorporates scoring.
- generate new auto research runs using the highest scores (but do it a bit fuzzy so we try to avoid local maximum)

Repeat for days and see if you get better images.
