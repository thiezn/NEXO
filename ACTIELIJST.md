# Mijn nieuwe actielijst

Even afgeleid van MRPF (wilde alleen maar heeeel even een logootje genereren, vervolgens zit ik openclaw na te bouwen, image generation pipelines te bouwen, scummvm assets extraction, vision OS game aan het maken in rust en swift en epub image generator te fixen...).

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
