## Nexo gateway

The Nexo gateway is the central messaging system for WebSocket communication. It is designed to be decoupled from the brain, tools, and AI inference. The gateway's primary responsibility is to route incoming requests to the brain asynchronously, allowing the brain to manage synchronization.

### Brain of the system

The Nexo gateway acts as the brain and is the central component of the system. It operates synchronously in a 'loop' and manages the state of the system. This ensures that our distributed network of nodes and tools remains in sync.

There are three key sections to the brain:

- The loop: This is the main game loop that runs continuously, processing incoming messages and managing the state of the system. Multiple loops can run concurrently, but they are isolated from each other to ensure consistency.
- Storage: This is where the brain manages both in-memory and persistent storage. It can use SQL for structured data and Markdown for more fluid, unstructured data like providing context to the model. The brain is responsible for loading and saving this data as needed.
- Cron jobs: This is where the brain manages scheduled tasks and background processes. Cron jobs are stored in the database and can be triggered based on time or specific events. The brain ensures that these jobs are executed at the right time and manages their state.

The brain receives structured requests from the gateway websocket and invokes a loop for each new request.

### The loop

The agentic loop is the full 'real' run of the system:

- request intake
- context assembly
- model inference
- tool calls
- response generation / streaming replies
- persistent storage updates

It's the authoritive path that turns a message into actions and a final reply, whilst keeping session state consistent.

A loop is a single, serialized run per session that emits lifecycle and stream events as the model things, calls tools and streams output. 

The gateway keeps track of the available capabilities of the nodes and routes them to available ones. When it invokes a node, it needs to lock the capability in SQL to ensure other concurrent agent loops are not able to use the same capability until it's released. This is how we manage concurrency and ensure that the system remains consistent even when multiple agents are running simultaneously.

This requires careful coordination between the gateway and the brain to manage state and ensure that capabilities are correctly allocated and released.

### Sessions

A client can have multiple sessions and maintains a session id locally in it's RAM. It can create a new session by sending a message to the gateway, which will generate a new session id and return it to the client. The client can then use this session id for subsequent messages to maintain context across interactions.

If the client breaks or restarts, it can query the gateway for a list of active sessions and their associated context, allowing it to pick up where it left off. This is crucial for maintaining continuity in interactions, especially for long-running tasks or conversations.

A client can clear existing sessions by sending a message to the gateway, which will delete the session and its associated context from the system. This allows clients to manage their sessions and ensure that they are not holding onto unnecessary context or resources.

## Nexo-nodes

Nexo nodes can offer both AI models and tools. They are the only components that can do this. We can have multiple nodes, and they advertise their capabilities to the gateway. The gateway routes these capabilities and updates to the brain so it can load them into RAM.
