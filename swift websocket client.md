# Prompt Swift WebSocket Client

Build a WebSocket Client in Moretimer Swift App to interact with the NEXO Gateway. This client will be eventually responsible for sending tool calls to the gateway, receiving generated images, and handling user feedback (thumbs up/down) to improve the AI model.

For now we need to build the WebSocket client and the existing gateway protocol.

## Schema and architecture

We can generate the JSON Schema for the protocol using the rust `nexo-client schema` command. The schema will be generated on the fly directly from the Rust code definitions.

I already have generated the latest schema for you to review @nexo-schema.json

- Details on the architecture are described @docs/src/nexo/architecture.md
- Details of the gateway protocol are described @docs/src/nexo/gateway_protocol.md

## WebSocket Client

Make sure to use Apple's latest recommended WebSocket API which is the 'Network framework' using the 'NetworkConnection'. Do not use the older 'URLSession using URLSessionWebSocketTask'.

We should leverage all the latest features for Swift concurrency and async/await patterns to make the code clean and efficient. Using an AsyncSequence to handle incoming WebSocket messages would be ideal for processing the stream of events from the gateway.

Make sure to use the xcode mcp to retrieve documentation    on the Network Framework API and async best practices from Apple.

## Errors

Make sure to integrate the WebSocket client into our @Moretimer/Moretimer/Errors/ErrorManager.swift so that we can track connection issues, protocol errors, and other WebSocket related problems in a centralized way.

When connection issues occur, perform automatic retries with exponential backoff. When a certain threshold is reached, surface the error to the user with actionable guidance (e.g. "Unable to connect to gateway. Please check your network connection or try again later. Press 'here' to retry.").

## Folder Structure

Add the WebSocket client code in a new folder called 'NEXO' within the Swift app project, with subfolders for Models and Services. This will help keep the code organized and modular.

I've created a placeholder file at @Moretimer/Moretimer/NEXO/Services/NexoService.swift where the main client implementation will go.

## Propagate the WebSocket client through the app

Use the @AppContext.swift implementation to initialize the WebSocket client and make it available throughout the app. This will allow us to easily send messages to the gateway and listen for responses from any view or component in the app.




Key components to build first:

- WebSocket interface between Swift and Rust.
- Tool call through websocket, generate image using this prompt
- After completion, send image back to Swift thread
- Prompt user to give thumbs up or down
- Store thumbs up in SwiftData, and send thumbs up/down data back to Rust for storage

Follow up focus:
- Keep image model loaded in memory so we can generate images faster. Need interface from websocket gateway into AI generation.

