# TurboQuant

My initial prompt:

Google just came out with an article and paper on a new algorithm called TurboQuant that could significantly improve KV cache memory consumption/lookup.

Read the article here: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/?utm_source=twitter&utm_medium=social&utm_campaign=social_post&utm_content=gr-acct
Read the paper in detail here: https://arxiv.org/pdf/2504.19874

# Goals

Help me understand what impact this could have for me to run local LLM models inference on Macbook Apple Silicon M4 Max 36GB.

I use Rust and Candle to implement various open models, the main one being:
- qwen3-4b-q5km
- qwen3-30b-a3b-q4km

I need to understand what needs to happen to replace the current algorithm used for KV cache in these models, with the new TurboQuant algorithm. How feasible is it to get a frontier Coding agent like OpenAI 5.4 to build an implementation of the algorithm in Rust, guided by the instructions in the paper.

- Make sure you deeply understand the paper. 
- Give me a clear plan of approach. Start with introduction, summary of the benefits, and Step by Step Implementation plan of all the building blocks needed to achieve the end result.


# Repo's to look at

Seems people are immediately trying to implement this. Copilot found some implementations and someone is currently building it :D

https://github.com/TheTom/turboquant_plus/commits/main/
