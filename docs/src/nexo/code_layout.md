

shared/nexo-ws-schema has the full websocket schema definition that both clients and gateway can use.
shared/nexo-tool has the trait definiton that tools should adopt to allow for integration into nexo
gateway is the main server component that clients connect to, it maintains connections to tools and agents and routes messages between them.
nexo-tools/* contains various tools. They will always be a rust binary/cli and library in one. They will adopt the NexoTool trait and implement the required methods. The cli can be used locally if you don't want to go through nexo. The library is mainly there to ensure decoupling the cli from any code that need them. Some tools might be dependent on the library of other tools, for instance the image processing tool will be used by the game extractor and the epub extractor, but also as a standalone tool for mutating images for other purposes.


TODO: Inspect https://github.com/zeroclaw-labs/zeroclaw/blob/41dd23175fe991adf1ee1a5693eff69e09ef2a3a/src/tools/traits.rs#L14 for example on how they do a generic trait. I like the name ToolSpec. Use the async_trait crate.

