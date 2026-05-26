- Examine the kv cache implementation in nexo ai. I think I need a proper abstraction for the KV cache regardless if a model uses candle or openai. However, it needs to integrate with the gateway as well as thats the full coordinator. Also the nexo-gateway doesn't have a router component as i initially planned so it's not really aligned with my architecture. Needs better review
- There is no routing component, we're actually not implementing anything there whilst its a key part of my multi distributed node pattern?


# The ChatTemplate, especially for OpenAI is completely messed up

- We're hardcoding the thinking and tool tokens in nexo-gateway in the system prompt. This is incorrect, we should instead use the system role for this. OpenAI REST API also has introduced the "developer" role, lets understand what this actually means.
- The OpenAI chat remote in nexo-ai is just pushing in whatever we're getting from the gateway.  It should instead use the proper chat template implemented in nexo-ai for gemma4 (and future other models). Our gateway should use our own generic struct for ChatMessage (or TranscriptMessage?) and the specific models should transpose it into the proper format expected.

NO I AM WRONG! OpenAI spec is actually opinionated and requires a specific format with role and context. the backend system is responsible for transforming it to the correct chat template. So does this mean our thinking tokens in nexo-gateway is wrong?!

# OpenAI specification

Requests:
"reasoning_tokens" is used by deepseek and perhaps gemma, openai spec returns this, need to think if i need it.

"reasoning_effort" is also an openai param which we're not incorporating at the moment


Responses:

the respone contains "choices" and "reasoning_content" and "usage". We don't use the reasoning_content at the moment.

They also support "stream" that emit reasoning tokens, I'd really love this I think as this will allow me to follow along with websockets if i'm not mistaken.


## The OpenAI spec is ok but I want nexo-spec to be able to deviate

Depending on the Role, the content and parameters are different. I think i'd either need to create something like this, or perhaps use a new-type state builder? Also, at the moment we're calling it a TranscriptMessage. Perhaps we should call it ChatMessage, and leave TranscriptMessageEntry (and perhaps strip the entry part?) for the persisted version with metadata?

The content should also be multimodal here, similar to the OpenAI spec. But OpenAiContent should be agnostic in nexo-spec, and we should just be able to convert to the OpenAI format, even if it is 100% the same at the moment. This way we can add our own deviations in nexo-spec if needed in the future.

```rust
#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String, // Strictly text
    },
    Developer {
        content: String, // Strictly text
    },
    User {
        content: OpenAiContent, // Fully multimodal (Text or Parts)
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>, // Usually text or omitted during tool calls
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: String, // Strictly stringified execution output
        tool_call_id: String,
    },
}
```

## UPDATE: I'm trying too hard myself. 

I should use mistral-rs (A lot of supported models) and mold-ai-inference (for flux + LoRA). This means I can kill the OpenAI servers.

Note, I should use mistral-rs-core library, and pull it directly from github. The crates.io seems to be a lot older.
