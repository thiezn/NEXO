# Temporary brainstorming


I need to collect my thoughts and write some pseudo code to try and understand the best pattern.


- Adding some hashsets now for activesession/model ids in nexo agents. This will need to move to storage layer using transactions for locking, recovering from failures/restarts and atomicity
- Parallel actions on a single session should never happen (Tool calling could be parallel but that would be marked as a single 'run_tools[Vec<ToolCall>]' action.)
- The Agent will pluck stuff from the queue and run it. This means it needs a loop, since we could
potentially handle multiple sessions at the same time, we'll need some timer to check for new sessions to handle. This could be a simple sleep or a more complex async wait on a channel that is notified when new sessions are added to the queue, but need understanding of how the queue is stored.
- At the moment we've modelled the inferencerequest between user -> gateway, and gateway -> node with the same inference request. This is wrong as we'd need to add system prompt and have picked a model already when gw->node. This will make it a lot more clear to reason about and enforce things. the Node should not need to know about model selection, the user should not need to know about system prompt or specific models.
- Still undecided if NexoAgent will be a single instance, or if we will spawn multiple instances. Remember to replace &self with self if its a single instance so we don't accidently break that contract. I'm inclined now to have NexoAgent be single instance. Internally it could still use threading to run multiple sessions in parallel. 


## Some initial rambling on the run function

// 0. TODO: How should we handle the queue. We'd either keep run here private and only allow
        // NexoGateway to push items on the queue, or we handle the queuing here. Or perhaps NexoAgent should be a
        // fresh instance per run? I think I want to block actions being processed by the NexoAgent to avoid concurrency
        // issues with the database and model state, so perhaps a single instance is best. We do want to support multi-user
        // so that could lead to constant load/unload of models but I think i want to accept that. its where the router will
        // eventually come in and perhaps scaling compute with multiple nodes. I'm leaning to a single instance that processes
        // one incoming event at a time. An event could be a new inference request, or a cron job, a nexo-node load model state change, etc.
        // We need to ensure that this loop can then finish every task quickly, so that means we need to treat gateway tool calling
        // as an external action that also reports back an event to the NexoAgent when completed. So, a separate NexoAgentMsg input/output enum
        // will allow us to handle events from different part of the system. Similar to NodeToGateway, etc this should embed
        // nexo-core types like InferenceRequest to avoid duplication of types. Still there will be a lot of similar types (NexoAgentMsg vs NodeToGateway) but
        // it will strongly type / decouple the different parts of the system. Here we also need to think if we need multiple mpsc channels for different
        // systems like interfacing with the NexoGateway or gateway Tool calls. Perhaps not, perhaps NexoGateway should do the gateway tool calls as well as the websocket
        // parts? Yes, that sounds right.

        // Roughly, the loop will be: (TODO: Should the agent loop until a full single run is completed, or should we be able to load balance multiple run rounds in parallel)?
        // I guess theoretically a run could take a LOT of time, so that would lead to load-balancing choice already? So that means a loop is a single round?
        // Should I call a tool call, a summary action, etc, a round as well?
        // 1. run router, this will determine which model we want to use, if there's a node available already with that model or trigger a load of a model.
        // 2. run context_manager. This manages system_prompt, create/retrieve session so far, check if compaction is needed, etc. The system prompt can be model specific so it's important to know what model already.
        // 3. run inference on the model. This will start a new run 1 and round 1. We will stream back responses, and after a full inference round is completed, follow up with required actions like tool call or new inference rounds. When the model returns content with no tool calls the inference run is completed. During a run, we also might have to act on interruptions from the client, e.g. provide additional context, or stop the inference run.

        // Every step here requires some storage in the gateway SQL table. We capture this is a session table which stores everything, including summarization steps, tool calls, tool responses, model outputs, model inputs, etc. This allows us to build a full sequential view of a session. This will act as an append only log. We will implement forking of sessions, allowing us to hook into earlier context to start new sessions going in different directions.

        // Parallelization, where is the boundary? We probably want to lock paralellization on the session boundary? This way we can serve
        // multiple clients OR multiple sessions of a single client in parallel. Depending on model capacity, parallelization could still be queued
        // but thats a separate mechanism. So a NexoAgent instance should be bound to a single session? This will allow us to enforce
        // the append-only log of a session.
