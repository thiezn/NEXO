
Chat sessions at the moment are now limited to feeding in the full conversation of

user: hi
assistant: hello
user: how are you?
assistant: I'm good, thanks for asking!


This uses chat templating system, and each model might have a slightly different approach. Key concept is that for EVERY inference prompt, we feed in the FULL conversation, of both actors. KV Cache ensures we don't have to recalculate the previously typed text, it only needs to calculate the relationship between the new token and all the old tokens.

This is great and all but only has TWO actors, the user and the assistant.

What about creating a conversation between multiple actors? Is there an existing templating system out there that supports this? Does it actually make sense?

I can imagine that its good for a model to understand WHO said what, so it can better understand the context of what is stated. This could open up some very interesting use cases. Think of a bot that listens in on an MSTeams chat and understands each actor in the chat, and can perform actions based on that.


- Perhaps this can be solved with a prompting strategy in nexo-gateway? Clients send their unique ID along with their message. I can then transform incoming client requests into a conversational format.

E.g User1 says "hi", I transform that into "User1: hi". User2 says "hello", I transform that into "User2: hello". Then I feed the entire conversation into the model, and it can understand who said what. Very simple, but probably effective enough? The system prompt for the session could contain initial instructions and identify all the users that are in the conversation, so the model can understand the context of each user from the start.
