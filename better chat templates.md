
I need to completely abstract away the special tokens for the model. 

A generic model trait for Multimodal or Text, etc needs to have optional think tokens, chanel tokens, image tokens, etc. The gateway shoudnn't manually construct them as it's doing now in the loop_runner.rs.


We rather have to have an abstraction layer that we pass on to nexo-node, an it in turn passes the abstraction to nexo-ai. Nexo-ai can then construct the real eventual prompt for the model. The general abstraction should be in @shared/nexo-spec/src/model.rs and the implementation of the abstraction should be in @nexo-ai/src/models/...


https://unsloth.ai/docs/models/gemma-4


They refer to this as chat templates so probably thats the right abstraction. Each Chat/Tool/Multimodal model should have it's own chat template implementation.


## Asked GPT5.4 this

I am implementing various open weight chat, tool and multimodal LLMs in Rust.

I have to create a good abstraction for the ChatTemplate, so I'm able to swap out models easily and keep things working.

There are a lot of different chat templates out there which makes this a bit more complicated. But there are some commonalities like prompt tokens thinking, tool calling, the various roles (system, user, assistant), and output tokens like <channel|>, <turn|> and <final_answer>

The names of these tokens, and the support of these tokens could vary per model. We need a good generic trait/struct we can use to capture incoming prompts from clients (through websockets json), which eventually end up in rust code that calls a specific model implementation. At that point the incoming generic thing needs to be translated to the correct format.

If a prompt is passed with some token/feature types not supported, the specific implementation should give an error.

Search for commonalities between chat templates, think about a good naming scheme and produce a good rust pattern I can use to accomplish my goal.

### Answer

A solid way to approach this is to treat **“chat template” as a compilation target**, not as your canonical in-memory format.

In practice, the ecosystems that matter here all converge on the same idea:

*   a chat request starts as a **sequence of structured messages** (`role` + `content`) rather than raw prompt text, and the template converts that structure into the exact token/text layout expected by the model. Hugging Face explicitly describes chat templates this way, and OpenAI-style chat APIs do the same at the API layer. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating), [\[developers...openai.com\]](https://developers.openai.com/api/reference/chat-completions/overview)
*   the most common roles are `system`, `user`, and `assistant`, but tool-related turns and multimodal content are also first-class in modern systems. HF uses `role` + `content`; OpenAI-style tool calling introduces `tools` and `tool_calls`; multimodal models represent `content` as a list of typed blocks rather than a plain string. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating), [\[developers...openai.com\]](https://developers.openai.com/api/docs/guides/function-calling), [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating_multimodal)
*   the *surface syntax* varies a lot: some models use ChatML markers like `<|im_start|>assistant`, some use `[INST]...[/INST]`, some embed the first system prompt inside the first user turn, and tool calling may be expressed via JSON, XML-ish wrappers, or model-specific sentinel tokens. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating), [\[reddit.com\]](https://www.reddit.com/r/LocalLLaMA/comments/17bksmc/official_hugging_face_chat_templates/), [\[github.com\]](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
*   Jinja-style chat templates have become the de facto interchange format across the open-weight stack: HF stores templates in tokenizer metadata, `llama.cpp` uses `minja` to support HF-style templates, and SGLang can load either JSON-format templates or Hugging Face Jinja templates. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating), [\[github.com\]](https://github.com/google/minja), [\[docs.sglang.io\]](https://docs.sglang.io/references/custom_chat_template.html)

That means the right abstraction is **not** “a struct containing literal tokens like `<|turn|>` or `<final_answer>`”, but rather:

1.  a **semantic IR** (intermediate representation) for client requests,
2.  a **capability model** for each dialect/model family, and
3.  a **compiler/parser pair** that maps between semantic IR and model-specific prompt/output text.

***

# Recommended naming scheme

The biggest design mistake would be to let model-specific token names leak into your generic API. I would name things by **semantics**, not by token text.

## Use semantic names like these

### For request-side concepts

*   `Conversation`
*   `Message`
*   `Role`
*   `ContentPart`
*   `ToolSpec`
*   `ToolChoice`
*   `ResponsePrefill`
*   `AssistantContinuation`
*   `GenerationCue`

### For output-side concepts

*   `AssistantEvent`
*   `TextDelta`
*   `ReasoningDelta`
*   `ToolCallDelta`
*   `ToolResult`
*   `FinalAnswer`
*   `Channel`
*   `StructuredSegment`

### For template/model metadata

*   `ChatDialect`
*   `DialectCapabilities`
*   `RenderOptions`
*   `RenderResult`
*   `OutputParser`
*   `TemplateCompiler`
*   `FeatureSupport`
*   `UnsupportedFeature`

## Avoid these in the generic layer

Avoid generic field names like:

*   `im_start_token`
*   `final_answer_token`
*   `turn_token`
*   `assistant_prefix_token`

Those belong in a **dialect implementation**, not in the public request model.

Instead, if you need token mappings, make them semantic:

```rust
enum MarkerKind {
    BeginMessage,
    EndMessage,
    BeginAssistantTurn,
    BeginReasoning,
    EndReasoning,
    BeginToolCalls,
    EndToolCalls,
    BeginFinalAnswer,
    EndFinalAnswer,
    ChannelTag,
}
```

and then let a model dialect decide whether `BeginFinalAnswer` is:

*   unsupported,
*   represented by `<final_answer>`,
*   represented by `<|channel|>final`,
*   or represented by nothing at all.

That keeps your client contract stable even when model syntax changes.

***

# The pattern I recommend

## 1) Canonical semantic IR

Make your websocket JSON deserialize into a **provider-neutral AST**.

```rust
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)]
    pub response: ResponseConfig,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>, // assistant-emitted calls already in history
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { url: Option<String>, bytes_b64: Option<String> },
    Audio { url: Option<String>, bytes_b64: Option<String> },
    Video { url: Option<String>, bytes_b64: Option<String> },
    File  { name: String, mime: Option<String>, bytes_b64: String },
    ReasoningHint { text: String }, // optional semantic hint, not guaranteed renderable
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value, // JSON Schema
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Named { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseConfig {
    #[serde(default)]
    pub generation_cue: GenerationCue,
    #[serde(default)]
    pub prefill: Option<String>,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    #[serde(default)]
    pub requested_channels: Vec<Channel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GenerationCue {
    #[default]
    Auto,
    StartAssistantTurn,
    ContinueLastAssistantMessage,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub request_visible_reasoning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Reasoning,
    Answer,
    ToolCalls,
    Commentary,
    Final,
}
```

### Why this shape works

*   It matches the real common denominator: message history + typed content + tools. HF’s text chat and multimodal docs both fit this shape; OpenAI-style tool calling also fits this shape. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating), [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating_multimodal), [\[developers...openai.com\]](https://developers.openai.com/api/docs/guides/function-calling)
*   It does **not** assume every model supports every feature.
*   It allows client JSON to stay stable even when the target model changes.

***

## 2) A capability model that is explicit

Every model adapter should publish capabilities up front, and validation should happen **before rendering**.

```rust
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct DialectCapabilities {
    pub roles: BTreeSet<Role>,
    pub input_modalities: BTreeSet<Modality>,
    pub tool_calling: ToolCallingSupport,
    pub reasoning: ReasoningSupport,
    pub generation_cue: GenerationCueSupport,
    pub channels: ChannelSupport,
    pub assistant_prefill: bool,
    pub parallel_tool_calls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    File,
}

#[derive(Debug, Clone)]
pub enum ToolCallingSupport {
    None,
    PromptedText,   // template asks for a text format; app parses it
    StructuredJson, // native-ish JSON or schema-constrained output
    Native,         // dialect has native parser/formatter
}

#[derive(Debug, Clone)]
pub enum ReasoningSupport {
    None,
    HiddenOnly,
    VisibleTagged,
    VisibleChannelized,
}

#[derive(Debug, Clone)]
pub enum GenerationCueSupport {
    None,
    NewAssistantTurn,
    ContinueLastAssistant,
    Both,
}

#[derive(Debug, Clone)]
pub enum ChannelSupport {
    None,
    Fixed(BTreeSet<Channel>),
    Arbitrary,
}
```

This is important because tool calling and reasoning are **not uniformly supported**. For example, llama.cpp explicitly distinguishes native handlers from generic tool-call handling, and vLLM documents model/parser-specific tool-calling support rather than a universal one-size-fits-all mechanism. [\[github.com\]](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md), [\[docs.vllm.ai\]](https://docs.vllm.ai/en/latest/features/tool_calling/)

***

## 3) Split “template rendering” from “output parsing”

Do **not** make one mega-trait that does everything. Use two complementary traits.

```rust
pub trait ChatDialect: Send + Sync {
    fn id(&self) -> &'static str;
    fn capabilities(&self) -> &DialectCapabilities;

    fn validate(&self, req: &ChatRequest) -> Result<(), TemplateError>;

    fn render(
        &self,
        req: &ChatRequest,
        opts: &RenderOptions,
    ) -> Result<RenderResult, TemplateError>;

    fn parser(&self) -> Box<dyn OutputParser>;
}

pub trait OutputParser: Send {
    fn feed(&mut self, chunk: &str) -> Result<Vec<AssistantEvent>, TemplateError>;
    fn finish(&mut self) -> Result<Vec<AssistantEvent>, TemplateError>;
}

#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    pub include_bos: bool,
    pub include_eos: bool,
    pub add_generation_cue: bool,
}

#[derive(Debug, Clone)]
pub struct RenderResult {
    pub prompt: String,
    pub stop_sequences: Vec<String>,
    pub metadata: RenderMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct RenderMetadata {
    pub expected_channels: Vec<Channel>,
    pub parser_mode: ParserMode,
}

#[derive(Debug, Clone)]
pub enum ParserMode {
    PlainText,
    Tagged,
    JsonToolCalls,
    XmlToolCalls,
    Mixed,
}
```

### Why the split matters

Because formatting and parsing are separate concerns:

*   Formatting: structured messages → model-specific prompt
*   Parsing: raw model stream → structured events (`text`, `reasoning`, `tool_call`, `final_answer`, etc.)

`llama.cpp` is a nice proof point here: it has both template rendering and an “autoparser” / message parser layer for extracting tool calls and reasoning from model output. [\[deepwiki.com\]](https://deepwiki.com/ggml-org/llama.cpp/3.9-chat-templates-and-message-parsing), [\[github.com\]](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

***

## 4) Have two implementation styles: declarative + custom

This is the part that keeps the abstraction from collapsing under edge cases.

### A. `DeclarativeDialect`

For “normal” templates where rendering can be described by:

*   markers,
*   generation cue behavior,
*   stop sequences,
*   role wrappers,
*   tool schema insertion strategy,
*   output tags/channels.

### B. `CustomDialect`

For oddballs like:

*   Llama-2 style system-prompt folding into first user turn,
*   special multi-part tool protocols,
*   `<|channel|>` / `<final_answer>` style responses,
*   reasoning tags with bespoke parsing.

HF’s own history here is instructive: simple prefix/suffix systems turned out not to be expressive enough, which is exactly why Jinja templating became the chosen mechanism. [\[reddit.com\]](https://www.reddit.com/r/LocalLLaMA/comments/17bksmc/official_hugging_face_chat_templates/), [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating)

So in Rust:

```rust
pub enum DialectImpl {
    Declarative(DeclarativeDialect),
    Custom(Box<dyn ChatDialect>),
}
```

This gives you an 80/20 path:

*   most models: declarative data + small helpers,
*   weird models: handwritten adapter.

***

# Error strategy

If a request uses unsupported features, fail **during validation**, not halfway through generation.

```rust
#[derive(thiserror::Error, Debug)]
pub enum TemplateError {
    #[error("unsupported role {role:?} for dialect {dialect}")]
    UnsupportedRole { role: Role, dialect: String },

    #[error("unsupported modality {modality:?} for dialect {dialect}")]
    UnsupportedModality { modality: Modality, dialect: String },

    #[error("unsupported feature {feature} for dialect {dialect}")]
    UnsupportedFeature { feature: String, dialect: String },

    #[error("invalid request: {reason}")]
    InvalidRequest { reason: String },

    #[error("render failed: {reason}")]
    RenderFailed { reason: String },

    #[error("parse failed: {reason}")]
    ParseFailed { reason: String },
}
```

Examples:

*   image content passed to a text-only model → `UnsupportedModality`
*   `tool_choice = Required` on a model with `ToolCallingSupport::None` → `UnsupportedFeature`
*   `generation_cue = ContinueLastAssistantMessage` on a dialect that only supports new assistant turns → `UnsupportedFeature`

HF explicitly calls out that `add_generation_prompt` and `continue_final_message` are distinct behaviors and should not be combined; your validation layer should encode that kind of rule generically. [\[huggingface.co\]](https://huggingface.co/docs/transformers/chat_templating)

***

# The design principle I think will save you the most pain

## Treat tokens as *dialect internals*, not API

Your client should say:

> “Start a new assistant turn, tools are available, visible reasoning requested, answer channel preferred”

—not:

> “emit `<|assistant|><|channel|>analysis<|turn|>`”

That way:

*   one model may map “reasoning” to hidden scratchpad and reject visibility,
*   another may map it to `<think>...</think>`,
*   another may expose `<|channel|>analysis`,
*   another may not support it at all.

Same semantic request, different dialect behavior.

***

# A practical architecture

```rust
pub trait ModelAdapter {
    fn model_id(&self) -> &str;
    fn dialect(&self) -> &dyn ChatDialect;

    fn infer(&self, rendered: RenderResult) -> anyhow::Result<ModelStream>;
}
```

Flow:

1.  websocket JSON → `ChatRequest`
2.  pick `ModelAdapter`
3.  `adapter.dialect().validate(&req)?`
4.  `let rendered = adapter.dialect().render(&req, &opts)?`
5.  send `rendered.prompt` to model
6.  stream tokens/chunks through `adapter.dialect().parser()`
7.  emit normalized `AssistantEvent`s to the websocket client

This makes model swapping easy because the rest of the stack consumes **events**, not raw model text.

***

# My concrete recommendation in one sentence

Build around a **semantic conversation IR + explicit capability negotiation + dialect-specific renderer/parser pair**, with token names hidden behind dialect implementations.

If you want, I can take this one step further and give you a **production-ready Rust crate layout** (`chat_ir`, `dialects`, `parsers`, `model_adapters`) plus a **worked example** for:

*   ChatML,
*   Llama-2 `[INST]`,
*   a tool-calling model,
*   and a “reasoning + final answer channel” model.
