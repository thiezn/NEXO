# Mijn nieuwe actielijst

Even afgeleid van MRPF (wilde alleen maar heeeel even een logootje genereren, vervolgens zit ik openclaw na te bouwen, image generation pipelines te bouwen, scummvm assets extraction, vision OS game aan het maken in rust en swift en epub image generator te fixen...).

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
- gguf with lama apparently performs a lot better than candle on ML. Consider rebuilding using that.


## Ok, a bit of focus, I want to get something done.

Got a whole bunch of code but nothing tangible. What would i consider tangible?

- Generating images I want. (This will be for my epub idea and my game engine)
- Question and Answer flow working well from app to backend. (This will allow me to have anki cards and have image training feedback loop.)


Key components to build first:

- WebSocket interface between Swift and Rust.
- Tool call through websocket, generate image using this prompt
- After completion, send image back to Swift thread
- Prompt user to give thumbs up or down
- Store thumbs up in SwiftData, and send thumbs up/down data back to Rust for storage

Follow up focus:
- Keep image model loaded in memory so we can generate images faster. Need interface from websocket gateway into AI generation.


## Swift UI Bugs

- answer kaart achtergrond moet andere kleur zijn. 
- Scrollen door messages als er kaarten zijn moet poppen, voor normale messages moet hij gewoon smooth scrollen.
- text typen in anwer is minder fijn want je ziet de text dan niet. Hoe op te lossen?
- Als je een single answer geeft op een kaart, moet hij doorscrollen naar de volgende kaart.
- Gekke bug over de toolbar, zit denk ik een overlay op
- glasseffect is denk ik op toolbar item geplaats van m'n settings
- pagination reading view word text door elkaar gehaald
- Bij continuous reading zijn de chapter titels dubbel
- We moeten een boek vinden met wat mooiere opmaak. Er is CSS in de epub, misschien moeten we gewoon een webview gebruiken? Of is er een andere standaard in epub om css te parsen in een beter formaat? Probeer het javascript hacking boek eens te bekijken, die heeft van allerlei andere opmaak dingen volgens mij.
