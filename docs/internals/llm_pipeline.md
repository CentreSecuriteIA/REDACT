# llm_pipeline/: units of model calls, and how they are driven

`llm_pipeline/` sits between the stages and `llms/`. A stage writes each unit
of work as a generator that yields model calls; this layer drives many such
generators at once, batches or streams their calls, and reads replies to
decide the next call.

## Responsibilities and boundaries

It is responsible for:

- `LLMRequest`, the call a generator yields;
- the drivers: one generator (`drive_sync`), many in rounds
  (`drive_generators`), and a live pool fed from a lazy source
  (`drive_units`), with its async lane;
- the shared check rule (`is_accepted`) and the two retry steps a unit is
  built from (`checked`, `extracted`).

It must not:

- read or write files, ledgers or rows. A finished unit is handed to the
  caller's `on_done`; saving is the caller's job;
- turn a failed call into a result. A failed batch or async call propagates.
  Only an exception raised inside a unit's own generator goes to `on_error`;
- import anything but `redact.llms` and `redact.exceptions`.

## LLMRequest

```python
LLMRequest(model, messages, internals_id=None, *, client=None, max_tokens=None)
LLMRequest.for_client(client, messages, internals_id=None, max_tokens=None)
```

A request names a model; with `client` it goes to that client, else the
driver builds one with `ModelClient.create(model)`. Requests share a call
only when their reply limits agree. The jailbreak techniques yield requests
by model name; the content moderation units bind theirs to a client.

A unit is a generator:

```python
def unit(...):
    reply = yield LLMRequest.for_client(gen, messages)
    verdict = yield LLMRequest.for_client(check, check_messages(reply))
    return reply, is_accepted(verdict)
```

The driver sends each reply back with `.send(reply)` and takes the return
value when the generator stops.

## drive_units: the pool

```python
drive_units(source, *, on_done, on_error=None, pool_size, cap=None, resolve=None, progress=None) -> int
```

`source` yields `(key, generator)` pairs. The driver pulls them only as slots
free up, keeping at most `pool_size` units live, so a stage never builds all
its units up front. A unit that finishes is reported at once through
`on_done(key, value)`, so the stage can save it without waiting for the
others.

Each turn of the loop:

1. Admit units into free slots and start each one.
2. Hand back an async reply that has arrived, if any (see below), and go
   round again.
3. Group the pending requests of the other models by model, setup and reply
   limit.
4. Send every group that fills a call (`cap` requests, oldest first) as one
   `client.generate(...)`. When no group is full: while async requests are in
   flight, wait for the next reply, which may fill one; otherwise send the
   largest group, so a trickle still drains.
5. Send each reply back into its unit.

A unit that raises, or yields something other than an `LLMRequest`, goes to
`on_error(key, exc)`; without `on_error` it propagates. A `generate()` that
fails propagates whatever the unit.

`pool_size(batch_size, *clients)` gives the units a stage keeps live: twice
`batch_size`, or the widest async lane among the clients when that is larger.
Every stage on the pool calls `drive_units` with
`pool_size=pool_size(batch_size, ...)` and `cap=batch_size`, except the
constitution, which runs one unit at a time (`pool_size=1`).

## The async lane

A request whose client has `supports_async_calls` (vLLM and OpenAI-compatible
endpoints) is not grouped. It is sent on the shared event loop (`llms/aio.py`)
the moment its unit yields it, through `client.agenerate`, and its reply is
handed back the moment it arrives. No unit waits for a longer reply in the
same batch.

- Each model has at most `backend.max_workers` requests in flight, enforced
  by one semaphore per model and setup. For a local vLLM engine that is its
  `max_num_seqs` times its copies; for a hosted model its `workers`.
- A hosted model's rate limiter is charged per request.
- A `PromptTooLongError` from an async request is raised inside its unit, at
  the `yield` that made it. The unit may catch it; the stages turn it into a
  rejected row or ack. The other requests stay in flight.
- Any other failed async request propagates, as a failed `generate()` does.
- On any exit, a failure, an exception from `on_done`, or Ctrl-C, the requests
  still in flight are cancelled before `drive_units` returns. The wait for
  replies wakes every second, so Ctrl-C works on Windows too.
- With `progress`, the driver logs a model's first request on the lane (`at
  most N in flight`) and then a line every `cap` finished units.

Models without async calls (Anthropic, the introspection backend) take the
grouped path.

## The other drivers

| Function | Use |
|---|---|
| `drive_sync(gen, call)` | One generator, answered by a blocking `call(request) -> str`. Used by `combine_techniques`, `apply_combination`, `run_conversation` and the standalone description and seed calls |
| `drive_generators(gens, *, finalize, on_error=None, resolve=None, progress=None)` | Many generators, round by round: every pending request of a round is grouped by client and reply limit and sent as one call per group. Used by `optimization/search.py` |

## Checks and retry steps

| Function | Does |
|---|---|
| `is_accepted(reply)` | The one accept rule: the reply's first word is `yes`, `ok`, `okay`, `accept`, `accepted`, `pass` or `passed`, in any case, after leading whitespace, markdown (`*`, `_`, `#`, `>`, backticks) and quotes, curly ones included. So `**Yes**`, `Yes.` and `yes, because` accept; `Yesterday`, `Acceptable` and `Verdict: yes` reject |
| `check_sample(client, sample, build_check_messages, original="")` | One check call: `(accepted, reasoning)`, the reasoning being the full reply on rejection |
| `batch_check_samples(client, samples, build_check_messages, originals=None, batch_size=32)` | Many check calls, in chunks, in order |
| `checked(gen, build_gen_messages, check=None, build_check_messages=None, *, original, internals_ids, max_tokens, max_attempts=1)` | A unit step: generate, check, and generate again with the verdict as feedback, up to `max_attempts`. Returns `(text, accepted, reasoning)` |
| `extracted(gen, build_gen_messages, extract, *, internals_id, max_tokens, max_attempts=3)` | A unit step: generate until `extract(reply)` gives samples, up to `max_attempts`. Returns `(samples, attempts)` |

`build_check_messages` is `(original, sample) -> messages`, the shape the
checkers of `content_moderation/checker.py` build.

## How the stages use it

| Stage | Driver | Unit |
|---|---|---|
| constitution | `drive_units`, pool of 1 | `extracted` over one call |
| seeded inputs | `drive_units` | `extracted`, then one check request per sample |
| outputs | `drive_units` | `checked` |
| jailbreaks | `drive_units` | `jailbreak.chain.run_chain` over the unit's techniques |
| conversations | `drive_units` | `multi_turn.conversation_gen` |
| scoring | `drive_units` | one judge request |
| paraphrase | none: chunked `batch_generate_samples` and `batch_check_samples` | |
| standalone inputs | none: blocking calls and `batch_check_samples` | |
