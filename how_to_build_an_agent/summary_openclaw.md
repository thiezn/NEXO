Component 1: Gateway
What it does: Long-living WebSocket server (default: localhost:18789) that accepts inputs from any channel.
Key insight: “The Gateway never performs reasoning. It only routes messages. This keeps the system modular — if Slack goes down, WhatsApp still works.”
Channels: CLI, Slack, WhatsApp, Telegram, Discord, Gmail, SMS, smart home devices, custom webhooks


Component 2: Brain
What it does: Orchestrates LLM calls and runs the reasoning loop.
Key insight: “The Brain compiles a system prompt with available tools, sends it to the LLM, parses the response for tool calls, executes them, and loops until a final answer emerges.”
Model support: Claude, GPT, Ollama, vLLM, any OpenAI-compatible API


Component 3: Memory
What it does: Stores persistent context in local Markdown files.
Key insight: “Memory survives restarts, updates, and migrations. Everything under ~/.openclaw/memory/ is automatically loaded into context."
Default files: preferences.md, contacts.md, projects.md, learnings.md, tools.md


Component 4: Skills
What it does: Plug-in capabilities defined via Markdown + YAML.
Key insight: “Skills are how OpenClaw does things. Each skill has a manifest (skill.md) that defines triggers, permissions, and instructions. The agent calls skills during reasoning.”


ClawHub: Community marketplace with 5,700+ skills

Component 5: Heartbeat
What it does: Monitors tasks, schedules jobs, checks inboxes.
Key insight: “Heartbeat is what makes OpenClaw run 24/7. It can poll your email, check for new files, trigger scheduled reports — without you being online.”



