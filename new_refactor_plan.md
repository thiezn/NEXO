New refactor plan, asking for multi stage approach doesnt work as it likely does a whole lot more after the first step already that the split doesn't make sense. Think more of bite sized refactors.

Broadly I need to tackle the following blocks. I should open new fresh sessions for it and mention:

- old memory files are likely stale as we're really re-building
- Refer to old code, but don't make assumtions. Make it clean.
- Use the /rust skill


# nexo-ws-schema

use new nexo-core types for the ws schema, and remove any old spec related code.

# nexo-tools/*

Update to use new nexo-core


# nexo-ai

Lets build a small quick and dirty tui in nexo-ai so we can test if inference actually works

# nexo-node

Refactor nexo-node to use the new nexo-core and nexo-ai crates.

# nexo-gateway

Refactor nexo-gateway to use the new nexo-core and nexo-ai crates.

# Update documentation. 

This should be a full refactor after we've updated all the code.

