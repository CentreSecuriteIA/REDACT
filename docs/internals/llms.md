# llms/: models, backends, clients, prompts

`llms/` is the layer every model call goes through. Above it, no code
branches on which provider serves a model.

## Responsibilities and boundaries

`llms/` is responsible for:

- the model registry: which models exist, how each can be served, and the
  role each fills;
- backends: one configured model on one provider, with its defaults, rate
  budget and concurrency bound at construction;
- clients: a backend plus the way its batches are dispatched and rate
  limited;
- prompts: loading prompt files, rendering them into messages, and parsing
  replies into samples;
- the local GPU facts the planner needs (`resources/`) and the telemetry
  hook (`observe.py`).

It must not:

- loop over generate-check-regenerate, or decide what a verdict means for a
  unit: that is `llm_pipeline/`;
- know about a stage's files, ledgers or rows: that is `dataset/` and the
  stage cores;
- import anything above `llms/`. Telemetry leaves through `observe.record`,
  a hook the top level installs;
- branch on a backend's type. Callers read its capability flags.

## Modules

| Module | Holds |
|---|---|
| `model_config.py` | `ModelConfig`, `APIConfig`, `VLLMConfig`, `IntrospectConfig`; `MODEL_REGISTRY` (from `configs/llm/models.json`); `register_model`, `get_model_config`, `get_models_by_role`, `default_model_for_role`, `available_roles`, `model_compute_config` |
| `backends/base.py` | `LLMBackend`, `ComputeConfig`, the system-prompt helpers |
| `backends/capabilities.py` | `BACKEND_TYPES`, `API_BACKEND_TYPES`, `SETUP_TYPES`; `resolve_setup`, `transport_for`, `backend_for`, `validate_concurrency`, `compute_config_for`, `set_run_serving`, `clear_transport_caches` |
| `backends/openai.py` | `OpenAIBackend`: any OpenAI-compatible endpoint (Venice, OpenRouter) |
| `backends/anthropic.py` | `AnthropicBackend`: the Anthropic Messages API |
| `backends/vllm.py` | `VLLMBackend`: vLLM's `AsyncLLM` in this process |
| `backends/tokenizer_folder.py` | `tokenizer_folder`: the folder a vLLM engine loads its tokenizer from when `hf_config` replaces its config |
| `backends/introspection.py` | `TransformersIntrospectionBackend`: transformers, with internals capture |
| `client.py` | `ModelClient` |
| `wrappers.py` | `RateLimiter`, `shared_limiter`, `BatchCaller` |
| `router.py` | `generate_sample`, `batch_generate_samples`, `assert_single_sample_per_call` |
| `aio.py` | The one event loop that runs every async model call |
| `observe.py` | The telemetry hook: `record`, `set_emitter`, `set_stage`, `current_stage` |
| `progress.py` | `ProgressReporter`: throttled progress lines |
| `prompting/` | `prompts.py` (loading, overlay, rendering), `extraction.py` (parsing replies), `scaffold.py` (writing an editable prompt tree) |
| `resources/` | `estimate.py` (a model's GPU need from its HF config), `measure.py` (`nvidia-smi` probes) |

## The registry

A `ModelConfig` is a model's identity (name, roles, defaults) plus one or
more **setups**: `api`, `vllm`, `introspect`. Every config validates itself
when it is built, so the registry holds only coherent entries.
[configuration.md](../configuration.md#the-model-registry) lists every field.

The registry name is the model's stable identity: it keys roles, rate-limit
windows, telemetry and prices. `APIConfig.api_model_id` is what the provider
receives when its own id differs (`venice-uncensored` is sent as
`venice-uncensored-1-2`).

## Backends

A backend **is a configured model**. Everything fixed about one model on one
setup (name, `default_max_tokens`, `default_temperature`, system-prompt
policy, `rpm`, `max_workers`, provider parameters, sampling defaults) is
bound in `from_config(config)`. The calls take only what varies:

```python
backend.generate(messages_list, *, max_tokens=None, temperature=None, internals_ids=None, **kwargs) -> list[str]
await backend.agenerate(messages, *, max_tokens=None, temperature=None, internals_id=None, **kwargs) -> str
```

`generate` is batch-shaped: one sample is a batch of one. `agenerate` exists
on backends with `supports_async_calls`. System-role messages are split
out; an empty system prompt counts as none. When the model has
`supports_system_prompt: false` they are folded into the first user message
(the first message when none is the user's), joined by a blank line, and an
empty system message is dropped. `max_workers` is clamped to 1 at
construction when the class cannot take parallel calls.

Each class declares its capabilities as a class attribute, so they are read
without building a backend (which for a local one loads weights):

| Backend | `supports_native_batching` | `supports_parallel_calls` | `supports_internals` | `supports_async_calls` |
|---|---|---|---|---|
| `OpenAIBackend` | no | yes | no | yes |
| `AnthropicBackend` | no | no | no | no |
| `VLLMBackend` | yes | yes | no | yes |
| `TransformersIntrospectionBackend` | no | no | yes | no |

**What each caches.** A backend is cheap; what it uses may not be, so each
module caches that on its own key, with a lock against two loads at once:

| Resource | Key |
|---|---|
| OpenAI SDK clients, sync and async apart (connection pools) | `(base_url, api_key, max_retries, timeout_s)` |
| Anthropic SDK client | `(api_key, max_retries, timeout_s)` |
| vLLM engines | `(hf_model_id, quantization, engine kwargs)`: two registry entries with the same three share one load |
| transformers models | `(hf_model_id, device_map, torch_dtype, load kwargs)`; separate from vLLM's |

`clear_transport_caches()` clears them all. An SDK client is closed, and a
backend built before keeps it, so its next call fails. A local engine is shut
down and its memory freed, and a client built on it raises on its next call.

### VLLMBackend

- The engine is vLLM's `AsyncLLM`, in this process. Every item is its own
  engine request, awaited on the shared loop (`aio.py`); the engine batches
  whatever is in flight. `generate` sends one request per item and waits for
  all.
- Prompts go through the chat template of the engine's tokenizer, so any
  checkpoint whose tokenizer ships a usable `chat_template` works with no
  extra setting. For one that does not (Mistral-3.x), pass vLLM's own
  `vllm_kwargs={"tokenizer_mode": "mistral"}`; the library formats no
  prompt by hand.
- `max_workers` is the engine's `max_num_seqs` times its copies: what the
  async lane keeps in flight.
- The load uses the residency plan's `gpu_memory_utilization` and cards
  (`set_planned_utilization`, `set_planned_devices`), else 0.9 of the card.
  Without `max_model_len` the engine gets 10,000 tokens
  (`estimate.effective_max_model_len`), capped at the checkpoint's longest
  context: the HF config's `max_position_embeddings`, else the first context
  field `config.json` sets (read from the checkpoint folder or the hub cache,
  never downloaded); 10,000 when neither names one. Planned cards are set in `CUDA_VISIBLE_DEVICES` for
  the build, by UUID where `nvidia-smi` gives one; more cards than one copy
  spans become copies (`data_parallel_size`). A build with copies that
  fails is tried once more with one copy. Each engine records the share and
  cards it was built with; `engine_grants()` gives them for the engines
  loaded now.
- A setup with `hf_config` loads its tokenizer from a folder of its own:
  vLLM reads the `config.json` of the folder it loads the tokenizer from,
  not only `hf_config_path`, so the checkpoint's own config would still be
  parsed. `tokenizer_folder()` fetches only the snapshot's tokenizer and
  processor files (`TOKENIZER_FILES`; the hub cache serves them offline, and
  a local `hf_model_id` is read in place), links them, or copies them where
  the OS allows no link, beside a copy of the packaged `config.json`, and
  the engine gets the folder as `tokenizer` unless `vllm_kwargs` names one.
  The folder is `$HF_HOME/assets/redact/tokenizers/<checkpoint>-<hash>`,
  hashed from the snapshot's path and the config's bytes, so a new snapshot
  or a changed config gets a new one; one built before is reused. It is
  built at load, under a lock, through a temporary folder renamed into
  place. It stays out of the engine key, which already names the checkpoint
  and the config and which the planner computes without fetching anything.
- Defaults: `temperature` 0.7 when the model sets none; `top_p` 0.85 unless
  `VLLMConfig.sampling` says otherwise.
- Stats: an engine logs none (`disable_log_stats` true). With
  `"disable_log_stats": false` in `vllm_kwargs`, a task on the shared loop
  calls the engine's `do_log_stats()` every 10 s, so vLLM prints its stats
  line, the prefix-cache hit rate among them; a failed call is logged at
  DEBUG, and the task stops when the engine is released and at exit.
- A prompt longer than the context raises `PromptTooLongError` for that
  request only. A request aborted because its engine is released raises.
- Every request records one `call` event, a failed or cancelled one too
  (with `error`): when one request of a `generate()` fails, the others are
  cancelled and each records its event.
- A release calls the engine's `shutdown()`, collects garbage and empties
  torch's CUDA cache. vLLM may then log an `EngineDeadError`; it is the
  shutdown, not a failed call. An exit hook releases every engine, waiting
  for a load in progress at most 60 s before it ends the load's engine
  processes ([residency.md](residency.md#lifecycle)).

### OpenAIBackend and AnthropicBackend

- One SDK call per item. The SDK retries a failed call (a 429, a 5xx, a
  lost connection) `APIConfig.max_retries` times, 2 by default, with its own
  backoff, unseen by the rate limiter and the trace. `APIConfig.timeout_s`
  sets the seconds before the SDK gives up on a call; `None` keeps the SDK's
  default (10 minutes). A call that still fails stops the stage. Each call
  records one `call` event; an async call cancelled in flight records one
  too, with `error`.
- `OpenAIBackend` sends `default_extra_body` with each call that passes no
  `extra_body` of its own, and maps a context-length error to
  `PromptTooLongError`. It warns on an empty reply and on a reply stopped by
  the provider's content filter.
- `AnthropicBackend` sends the system prompt as Anthropic's `system`
  parameter, runs one call at a time, and maps a 400 that says the prompt
  is too long (or that it and `max_tokens` exceed the context) to
  `PromptTooLongError`. It warns on an empty reply.

### TransformersIntrospectionBackend

For research runs that need a model's internals: hidden states, attention,
logprobs. It is not a vLLM extension, since vLLM's continuous batching
discards the intermediate activations; it runs the checkpoint through
`transformers` instead, in place of vLLM for the run that needs the capture,
not beside a live vLLM engine of the same weights.

```python
from redact import generate_outputs
from redact.llms import IntrospectConfig, register_model

register_model(
    "my-model-introspect", backend_type="introspect",
    introspect=IntrospectConfig(
        hf_model_id="mistralai/Mistral-7B-v0.3",
        log_dir="./runs/internals",
        capture={"logprobs": True, "hidden_states": "last", "attention": False},
    ),
)
generate_outputs(model="my-model-introspect")   # captures, since the backend can
```

Each item is its own forward pass. An item with an `internals_id` is captured under
`<log_dir>/<internals_id>/`: always `meta.json` (settings, messages,
output), and per `capture` setting `logprobs.pt`, `hidden_states.pt`,
`attention.pt`. `rename_capture(old, new)` moves a capture once its final id
is known. Capture never adds a column or a row.

| Stage | Capture ids |
|---|---|
| outputs | `<input_id>/output`, `<input_id>/val_out`; later attempts under `/attempt_<n>` |
| seeded inputs | `<entry id>/<iteration>/input_<style>`; `<sample_id>/val_in_<style>` |
| paraphrase | `<input_id>/paraphrase/attempt_<iteration>`, renamed to `<input_id>/paraphrase/<sample_id>` |
| jailbreaks | `<input_id>/<label>/attempt_<iteration>/<technique>/<n>`, the root renamed to `<input_id>/<label>/<sample_id>` |

Conversations are not captured.

## Choosing a setup and building a backend

`resolve_setup(config, requested)` picks a setup, in order: the one
requested, the run's choice for the model (`set_run_serving`, from the
recipe), the entry's `backend_type`, the only setup present.
`transport_for` maps it to a key of `BACKEND_TYPES` (`api` becomes the
`APIConfig`'s own `backend_type`). `backend_for(config, setup)` builds the
backend, with the run's `workers` in place of `recommended_max_workers` when
the setup is an API one.
`validate_concurrency` refuses, at registration, more workers than a
series-only provider can use.

## ModelClient

```python
client = ModelClient.create("venice-uncensored")                       # hosted
client = ModelClient.create("venice-uncensored", backend_type="vllm")  # local weights
replies = client.generate(messages_list)
```

A client is a backend plus its dispatch, decided once from the backend's
flags:

- a native-batching backend (vLLM) gets the whole list, and the rate
  limiter, if any, is charged once per batch;
- any other goes through a `BatchCaller`: one call per item, on a thread
  pool of `max_workers`, or one after another when `max_workers` is 1.

`client.agenerate(messages)` makes one async call, charging the limiter per
call. `generate` raises when `internals_ids` reach a backend that cannot
capture. Clients are not cached: they are cheap, and what is expensive is
cached below them.

**A backend built by hand.** For an endpoint the registry does not describe,
build the backend and wrap it. A backend is one configured model, so the
model's name comes first:

```python
import os

from redact.llms import AnthropicBackend, ModelClient, OpenAIBackend, VLLMBackend

client = ModelClient(OpenAIBackend("my-model", api_key="...",
                                   base_url="https://api.example.com/v1", rpm=60))
client = ModelClient(AnthropicBackend("claude-opus-4-6",
                                      api_key=os.environ["ANTHROPIC_API_KEY"]))
client = ModelClient(VLLMBackend("my-local-model",
                                 hf_model_id="mistralai/Mistral-7B-v0.3"))  # loads the weights
```

Anything left out takes `LLMBackend`'s default, including `rpm=None`: no
rate limit at all. For a model you reuse, `register_model` it instead
([extending.md](../extending.md#add-a-model)): the setup is validated, and
`ModelClient.create(name)` wires it as it wires every shipped model.

**Rate limits.** `RateLimiter` enforces `rpm` over a sliding 60-second
window and is inert when the backend has no `rpm`, as every local setup has.
`shared_limiter(key)` hands out one window per key, so two clients of one
model share it. The key is the model name, or under
`rate_limit_scope: "endpoint"` the endpoint id (provider, URL, key
variable), shared by every model on that endpoint; those models must declare
the same `rpm`.

## Helpers

| Function | Does |
|---|---|
| `generate_sample(client, messages, **kw)` | One call, one reply |
| `batch_generate_samples(client, messages_list, batch_size=32, ...)` | Many replies, in chunks, in order |
| `assert_single_sample_per_call(client, n)` | Raises when internals capture meets several samples per call; no stage calls it, since every call of the inputs stage asks for one sample |
| `aio.submit(coro)`, `aio.run(coro)` | Schedule on the shared loop; run and block. The loop lives on one daemon thread, started on first use |
| `observe.record(event)` | Hand an event to the installed emitter; a no-op when none is; never raises |

## prompting/

| Function | Does |
|---|---|
| `resolve_prompt(pipeline, category, prompt_dir)` | The file a prompt resolves to, override first, and whether it is an override |
| `load_prompt(pipeline, category, prompt_dir)` | The prompt as a dict |
| `PromptTemplate.load(pipeline, category, *, prompt_dir=None, few_shot=True, format_style=None, **build_kwargs)` | A template: the system prompt rendered once, the user template on each call. With `format_style` the format instruction is appended |
| `build_messages(prompt_config, few_shot=True, format_style=None, **kwargs)` | One-shot render, the same kwargs for both parts |
| `get_format_instruction(style, num_samples, prompt_dir=None)` | The `numbered` or `structured_qa` instruction |
| `report_prompt_sources(prompt_dir)` | Logs overrides, and warns about override folders that match nothing |
| `extract_numbered_list`, `extract_structured_qa`, `extract_and_clean` | Parse a reply: a numbered list (`1. sample`, `2) sample`, `3- sample`, `4: sample`), or `**Question:** ... **Answer:** ...` pairs, optionally headed `**Prompt N:**`. `extract_numbered_list(text, strict=True)` rejects a reply that is not one clean list. With `expected_count`, `extract_structured_qa` and `extract_and_clean` reject a reply with another count whole: the result is empty |
| `clean_sample` | Strip markdown and stray chat tokens |
| `scaffold_prompt_tree(target_dir, source_dir=None, mode="copy")` | Write a copy (`copy`) or skeletons (`empty`) of the prompt tree |
| `scaffold_setup(target_dir, *, overwrite=False)` | Write a setup folder: the two example recipes with `"config_dir": "."`, their params files, the dataset type, taxonomy, augmentation, generation axes and dimensions files a `config_dir` overlays, and a `README.md` ([configuration.md](../configuration.md#your-own-config-folder)) |

A render error names the missing placeholder, and the prompt file when the
template comes from `PromptTemplate.load`.

## How the stages use it

- Fronts build clients with `ModelClient.create(name)`, the name defaulting
  to `default_model_for_role(role)`.
- Cores hand clients to `llm_pipeline`: units yield `LLMRequest`s bound to a
  client, or naming a model (`LLMRequest(model, messages)`, as the jailbreak
  techniques do), which the driver resolves with `ModelClient.create`.
- Prompts come from `load_prompt` or `PromptTemplate.load`; the checkers of
  `content_moderation/checker.py` are templates that build
  `(original, sample) -> messages`. `build_quality_checker(category,
  entry_type, subcategory)` judges an input against all three, so a benign
  or dual-use sample is not rejected for falling outside a harm category;
  `build_output_quality_checker(category, mode)` loads the checker of a
  response's answer mode (`output/quality_check/<mode>`), and
  `build_paraphrase_checker` the paraphrase check.
- The residency planner reads `MODEL_REGISTRY`, `resources/estimate.py` and
  `resources/measure.py`, and hands its grants to `backends/vllm.py`.
