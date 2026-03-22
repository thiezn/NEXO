# Mijn nieuwe actielijst

Even afgeleid van MRPF (wilde alleen maar heeeel even een logootje genereren, vervolgens zit ik openclaw na te bouwen, image generation pipelines te bouwen, scummvm assets extraction, vision OS game aan het maken in rust en swift en epub image generator te fixen...).

- Update the Swift UX skill to use progressive disclosre for the different components
- Review my epub exporter. It now included file paths AND base64 data. I want to change this to only generate one or the other.
- Hook in my raspberry pi with audio hat and make it a full flegded client to my gateway. Then it can launch tools, and the gateway can send it processed drum loops etc.
- Add shared helper crate for audio processing using Symphonia. We can use this in my speech to text/text to speech, text to beat generation tools. It could also become a separate tool that can be invoked by my gateway for changing audio files.
- Add helper crate of image transformations, similar to my audio transformations.
- Add Question and Answering cards to SwiftU
- Build shared lbirary for my websocket types. put them in a folder called schema, and make a folder hierarchy that resembles the nested json structure. Read up on the two different mod.rs styles of rust to see if the new one makes more organizational sense.
- Add console and indicatif to utl-helper crate and re-export them. This way we can keep the dependency shared between the different tools, and we can use indicatif for progress bars in the terminal for all of them.
