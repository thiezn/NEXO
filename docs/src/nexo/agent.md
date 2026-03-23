# Agent

The gateway runs a main agent loop. This is the heart of the system that actually executes tools.

I really want it to resemble a game engine loop, with a fixed tick rate and a clear separation between the "game state" (agent state) and the side-effecting operations (tool calls). This should make it easier to reason about the flow of data and the timing of operations. 

Not sure if thats a good idea, but I want a romantic view of the world.
