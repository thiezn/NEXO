# March 24 2026 - 20:30 - 21:30

Went ona. walk with Ani Grok to try and learn more about inference, how the concepts work.

## Key concepts.

I might be wrong in a few of these things but let me write quickly whilst i still sort of remember.

- Raw text is sent into the tokenizer.
- A tokeniser transforms the raw text into multiple tokens, each represented by a number, like 123
- Embeddings then transform those numbers into embedding vectors (tensors), basically some number thingie like 20,31,50 (Note that encoders are not used anymore, that was old stuff, now it's just embeddings)

**SOME NOTES GPT5.4 SAID ABOUT THIS above:**
    - 'during prompt processing (“prefill”), many prompt tokens are processed together;'**
    - Encoders and Decoders are still used, eg. for translation and summarization models.

- After this the embeddings are sent into the layers of the model
- A layer basically understands certain concepts like 'grammar' or something like that. Thiis is an emergent thing, we don't really fully understand what each layer does but this is a good way to reason about it.
- Each layer has a set of attention heads.
- An attention head has three components q, k and v (query, key and value).


- When a token comes in a normal (not a Mixture of Experts) layer, each attention head will use some match based on the query, key and value to determine how well of a match it has. The query is your vector (i think), the key and value is something thats in the weights or something. Various match happens.

**Another note by GPT** ```
    - key and value is not in the weights. weights contain the projection matrices that produce Q, K, and V from the current token representations; the keys and values themselves are runtime activations, not static objects sitting in the model.
```



- Each head (i think) does some normalization, and then it's determined what head has the highest match for you. It then routes your token through that to the next layer.

  **This part is the biggest misconception** ```
In normal multi-head attention, every token representation is processed by all heads in parallel, not by one selected head. The heads’ outputs are then merged/projected and added back into the running representation

If you’ve heard “routing,” that is more associated with Mixture-of-Experts-style components, not with ordinary attention-head selection. In mixture of experts, there basically is a thing that makes a decision about which heads to route the token through
  ```


- During this calculation step, the KV cache is updated. Basically the KV cache remembers the keys and values of the processed tokens so far. This is i think how 'context' emerges. Every time you push a new token through the model, the KV cache remembers what happened before and can steer you in the right direction. KV cache management is therefore important for context window management. A lot of models or inferene pipelines use a sliding window, basically when the cache is full it drops the oldest. Tools like Claude Code monitor the KV cache, and does a summarization step of everything in their cache, clears the caache and fills it with the summary. Its a great trick but also consumes tokens and compute, so its a tradeoff.
- Once the token comes out the layers, i think some more mathy stuff is happening like SoftMax, but i'm not sure
- Then at the end it picks a output token, the what comes next token prediction. There are settings that you as an model inference runner can tweak to determine the output of this. TI tihnk they call this temperature, high is loosy goosy, low is precise, they also talk about 'greedy' as in always pick the highest probabity token making it more rigid i gues but less suites for creativity. Sampling makes it more probability driven. This is really where you can play around with. I asked how companies like grok handle this. Grok offers some broat settings someone can set like, precise, funny, loose which sets some of these settings. Ani also mentioned something called 'User preference inference' where they do some light weight evaluation of your past prompts to see what you like. If you often laugh at jokes they optimize for that etc.



## GPT 5.4 summarization of how I could think about it

A cleaner mental model you can keep
Here’s a more accurate “engineer’s version” of your notes:

- Raw text goes to the tokenizer, which splits text into tokens/subwords and maps them to integer IDs. 
- The model turns token IDs into embedding vectors, and also adds/uses positional information so order is represented. 
- Those vectors pass through many transformer blocks. Each block typically has self-attention + feedforward/MLP (or related variant) with residual connections and normalization.
- In self-attention, each token representation is projected into Q, K, and V; attention scores are computed, softmax turns those scores into weights, and weighted values are combined. All heads run in parallel.
- During generation, the model stores prior K and V tensors in a KV cache so it does not recompute them for every new token.
- After the final layer, the model produces logits over the vocabulary, and decoding rules such as greedy, sampling, temperature, top-k, or top-p decide which token to emit next.
- That new token is appended, and the process repeats.


## A follow up question I asked around managing KV cache, context window and compaction

I am building my own local inference system running on Macbook M4 Max 36GB and use rust and candle to load a model, eg. gemma3, in memory and run inference through it.

I'm not sure what KV cache I'm using there and what candle, or my implentation exactly does with this. Will it likely keep growing the KV cache until memory runs out?

I want to be able to control this a bit with a separate process running in my inference code. Monitor how large the KV cache is, or how long ago a prompt came in and then clear the cache, or summarize the full generated tokens into a short compaction, clear the cache, warm up the cache again with the generated summary, ready for the next prompt
