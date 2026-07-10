Next prompt.

- Create the type state pattern for the Inference run. 
- We should ensure that we codify the state machine/order of things in the type pattern to ensure we cannot transition between invalid states. 
- This also then needs to work with the database.
- Certain states will be happening based on incoming events. We need to somehow glue this + the db state together to make everything type safe.
- Can we create a more robust db function calling structure that takes in these new types and transform them to the proper database store to avoid having all these optional properties on the generic update_run, and other similar functions.
