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
- turn a failed call into a result. A failed batch or async call propagates,
  except an async call's `PromptTooLongError`, which is raised inside its
  unit. Only an exception raised inside a unit's own generator goes to
  `on_error`;
- import anything but `redact.llms` and `redact.exceptions`.

## LLMRequest

```python
LLMRequest(model, messages, internals_id=None, *, client=None, max_tokens=None,
           temperature=None)
LLMRequest.for_client(client, messages, *, internals_id=None, max_tokens=None,
                      temperature=None)
```

A request names a model; with `client` it goes to that client, else the
driver builds one with `ModelClient.create(model)`. `max_tokens` and
`temperature` are the call's reply limit and sampling temperature, `None`
the model's defaults. Requests share a call only when their reply limits and
temperatures agree. The jailbreak techniques yield requests
by model name; the content moderation units bind theirs to a client.

A unit is a generator:

```python
def unit(gen, check, messages, check_messages):
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
3. Group the pending requests of the other models by model, setup, reply
   limit and temperature.
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
  `max_num_seqs` times its copies; for a hosted model its
  `recommended_max_workers`, or the recipe's `workers` for the run.
- A hosted model's rate limiter is charged per request.
- A `PromptTooLongError` from an async request is raised inside its unit, at
  the `yield` that made it; the other requests stay in flight. The inputs
  stage acks that entry rejected, outputs writes a rejected row, jailbreaks
  and conversations an error row, the combination search a failed candidate
  row, paraphrase acks the unit `prompt_too_long`, and scoring scores it
  false. The
  constitution and benign units take no `on_error`, so there it stops the
  stage.
- Any other failed async request propagates, as a failed `generate()` does.
- On any exit, a failure, an exception from `on_done`, or Ctrl-C, the requests
  still in flight are cancelled before `drive_units` returns. The wait for
  replies wakes every second, so Ctrl-C works on Windows too.
- With `progress`, the driver logs a model's first request on the lane (`at
  most N in flight`) and then a line every `cap` finished units.

Models without async calls (Anthropic, the introspection backend) take the
grouped path. There a `PromptTooLongError` fails the whole `generate()`,
which is not known to belong to one item, so it propagates and stops the
stage, as any failed call does. A stage's handling of a too-long prompt (a
rejected row, an ack, an error row) therefore applies on the async lane
only.

## The other drivers

| Function | Use |
|---|---|
| `drive_sync(gen, call)` | One generator, answered by a blocking `call(request) -> str`. Used by `combine_techniques`, `apply_combination` and `run_conversation` |
| `drive_generators(gens, *, finalize, on_error=None, resolve=None, progress=None)` | Many generators, round by round: every pending request of a round is grouped by client, reply limit and temperature and sent as one call per group. Used by `optimization/search.py` |

## Checks and retry steps

| Function | Does |
|---|---|
| `is_accepted(reply)` | The one accept rule: the reply's first word is `yes`, `ok`, `okay`, `accept`, `accepted`, `pass` or `passed`, in any case, after leading whitespace, markdown (`*`, `_`, `#`, `>`, backticks) and quotes, curly ones included. So `**Yes**`, `Yes.` and `yes, because` accept; `Yesterday`, `Acceptable` and `Verdict: yes` reject |
| `check_sample(client, sample, build_check_messages, original="")` | One check call: `(accepted, reasoning)`, the reasoning being the full reply on rejection |
| `batch_check_samples(client, samples, build_check_messages, originals=None, batch_size=32)` | Many check calls, in chunks, in order |
| `checked(gen, build_gen_messages, check=None, build_check_messages=None, *, original="", internals_ids=(None, None), max_tokens=(None, None), max_attempts=1, precheck=None, temperatures=(None, None))` | A unit step: generate, check, and generate again with the verdict as feedback, up to `max_attempts`. `gen` and `check` are each a client or a model name, which the driver resolves (a jailbreak technique's, as the typo check's). `max_tokens` and `temperatures` go to the (generation, check) requests. `precheck(text) -> reason`, with a check, is a code-side verdict before it: a non-empty reason rejects the reply with no check request, and is the feedback (or the last attempt's reasoning). Returns `(text, accepted, reasoning)` |
| `extracted(gen, build_gen_messages, extract, *, internals_id=None, max_tokens=None, max_attempts=3)` | A unit step: generate until `extract(reply)` gives samples, up to `max_attempts`. Returns `(samples, attempts)`. `gen` is a client, or a model name the driver resolves, as a jailbreak technique's requests carry it (the cognitive and persona scenarios) |

`build_check_messages` is `(original, sample) -> messages`, the shape the
checkers of `content_moderation/checker.py` build.

`CHECK_TEMPERATURE` (0.15) is the temperature the stages' check requests
ask for (inputs, outputs, paraphrase, and the jailbreak stage's checks of a
model's rewrite, scenario or benign pair); the generations keep the model's
default. A model registered with
`supports_temperature` false (one that rejects sampling settings, such as
`claude-opus-5-5`) is sent no temperature, the check's included
([llms.md](llms.md#the-registry)).

## How the stages use it

| Stage | Driver | `pool_size` | `cap` | Unit |
|---|---|---|---|---|
| constitution | `drive_units` | 1 | 1 | `extracted` over one call |
| inputs | `drive_units` | `pool_size(batch_size, gen, check)` | `batch_size` | `extracted` over one call for one sample, then one check request |
| outputs | `drive_units` | `pool_size(batch_size, client, check)` | `batch_size` | `checked` |
| jailbreaks | `drive_units` | `pool_size(batch_size, client, check)` | `batch_size` | `jailbreak.chain.run_chain` over the unit's techniques |
| benign data (`jailbreak/manipulation/benign.py`) | `drive_units` | `pool_size(batch_size, client, check)` | `batch_size` | `extracted` over one call: one subcategory and answer type, 3 attempts; then each pair's check request, and for a rejected pair `checked` over one pair asked for again, 2 attempts |
| combination search (`jailbreak/search.py`) | `drive_units` | `pool_size(batch_size, *clients)` | `batch_size` | one candidate combination: its `run_chain` calls, one request to the target, one to the judge |
| conversations | `drive_units` | `pool_size(batch_size, *clients)` | `batch_size` | `multi_turn.conversation_gen` |
| scoring | `drive_units` | `pool_size(batch_size, judge)` | `batch_size` | one judge request |
| paraphrase | `drive_units`, per paraphraser | `pool_size(batch_size, client, check)` | `batch_size` | `checked` with the gates as `precheck`, the reply cap per unit as its `max_tokens` |

The benign data runs with the jailbreak stage's `batch_size`.
