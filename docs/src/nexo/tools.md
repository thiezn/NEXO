# Tools

NEXO is designed around tool use. It provides a few out of the box default tools, and has an in memory registry of other tools it can leverage.

## Base Categories

These categories are the base tools provided directly by the gateway. They are tools that interact with the core of the NEXO system itself and provide fundamental capabilities that other tools and agents can build on top of.

### Core tools

- Read, Write, Delete files, Execute bash commands

### Memory tools

- Store, recall, forget

### Schedule tools

- cron_add, cron_list, etc

## Other tools

Not sure how to categorize this, but these should be the tools i can bring in and outof the system like my llm models, graphic tools, game extractor, epub extractor, etc. 

## Tool registry

The tool registry can be queried by clients to see which tools are currently available. It shows the status of the tools, and which llm model(s) its currently leveraging if applicable.

Nodes will be able to register with the gateway to provide additional tools. This all happens through WebSocket interface and health polling.
