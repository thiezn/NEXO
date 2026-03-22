# NEXO

NEXO - Neural Extension Operator

The end-all tool for myself to work on my projects.

Not sure yet what i'm going to use to build but I know rougly what I want. A sort of OpenClaw that will work on my projects independently on my macbook M1 POro Max64gb.


Features:

- Coordinate tasks to the right models (local models where it can, paid bigger models where it needs to)
- Invoke various of my repeatable tools
- Collect and organize my ideas and notes
- Interact with the Agent from my iPhone and MacBook using threads. A thread can be a project, a topic, a task, etc. It will be similar to a Claude session and get me to instruct the agent to do things or figure things out. It can open new threads with me if it needs to ask me something, i can open new ones if i need to ask it something, etc. It will be a way to have a conversation with the agent and keep track of it in an organized way.

## The Gateway

The gateway is the main entry point for the system. It will be a Rust websocket server that will interact with my Swift Apps and other clients. It will receive instructions from the clients, process them, hands off tools, checks cron, etc. It will be doing the game loop!

## Tools

Tools are things my model can invoke to do things. They can be local scripts, web tools, or even other models. The idea is that the model can decide which tool to use based on the task at hand and invoke it with the right parameters.

- A autoresearch skill. This will provide the scaffolding to create a new autoresearch project like train/improve a LoRA model, improve my codebase, anything else that has clear verifiable steps.
- A web retrieval tool. Could perhaps leverage mrpf engine for faster retrieval, but initial iteration just uses reqwests to retrieve things, integrates an ad-blocker etc, and feeds the content through markdownify to get the text content out of it. 
- Start a thread/conversation?
- Start a new LoRA training project (eg. feed in a set of images of a person and get it to train using a certain style)
- Start a new MusicMan LoRA training project
- Add an idea to the idea vault
- Summarize a set of ideas/notes
- Translate voice to text (whisper)
- Translate text to voice (TTS)
- Generate spritekit animations that can be imported into my Swift Projects
- Extract a podcast episode audio file
- Extract a youtube video audio file
- Extract a youtube video transcript
- Translate text
- Generate image(s) based on a chapter of a book or a story
- Read/exec/edit/write files
- Convert PDF and Epub to markdown

## Some of the infrastructure components

TODO: Investigate the OpenClaw architecture, and other rust based similar tools. Perhaps we should find the best/cleanest looking architecture, clone it and rip it to pieces and rebuild from there.

- A Rust websocket daemon that interacts with My Swift Apps.
- A trait for tools that can be invoked by the model. Each tool will implement this trait and the daemon will be able to invoke them based on the model's instructions.
- A memory system
- A model coordinator that decides which model to use for each task and coordinates the invocation of tools and the flow of information between them.
- A 'game' loop
- A prioritization system for tasks and threads
- A good system with re-usable crates and swift packages to quickly build new ad-hoc tools. Think of cli_helpers, SwiftUX helpers, API Clients.
- A good storage system, perhaps Sqlite + rust sqlx is great as we can build integration tests locally easily. Perhaps even move Swift App to SQLlite storage and ditch SwiftData to more easily share primitives/caching between the two. I will then miss out on some of the Swift Data helpers so might not be best but will think on it.
- A cron system


## OpenClaw's architecture

- Gateway
- Clients


## What Doesn't it do?

I think I should keep the programming stuff outside of it a bit. At least the hardcore building should probably stay in Claude Code as they will go a lot faster than I will. It should however be able to invoke bash and python to figure out stuff.


## How shall I tackle this? I've got a lot of projects going on at the same time and want to actuallyh aave some fun things to show off

- I want to get the image generation pipeline working
- Then I want to be able to run it on my Macbook M1 Max 64gb.
- Then I will start with the Apple Swift App for Rowan, Marly and Myself. For this I do already want to establish a few baseline Swift UX packages so I can more easily launch new tools and apps. For instance I've already started with my Vision Pro game, which is nice but lets make something a bit more tanglible like my epub to markdown extractor + Image generator. This is something very cool that could help Marly and make my wife impressed.
- Then I will add the feature for Rowan's anki cards. Take a picture of homework, extract the text and topics and spit out a prompt that I can feed into some model to generate questions and answers for anki cards. This is something that would be really cool and tangible to show off.
- Then implement the screen time token earnings in the app so Marly and Rowan can earn tokens by doing homework and reading.
- After this reflect on it and see if I can build a few of the above tools. Later on i can think about how to integrate them, as long as i have a library/api around it, this is trivial. Don't need to have the whole orchestator, first get my tools in order.


Then stop and sit and think a bit about what I want. I feel I'm getting very distracted from the MRPF engine, but maybe thats ok for now. Remember, money is obviously cool, but I want to learn new things and build cool stuff. Thats what really motivates me. I have a fear of pushing through with bug bounty hunting and thats just fine. You ARE learning things and still working out etc. Keep reminding yourself that you also pay attention to others and leave the thing. Once i've got my idea summarization thing going I can probabvly clear my mind a bit easier with these kind of things.
