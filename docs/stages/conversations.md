# Stages: conversations and scoring

Fronts and cores `generate_conversations` (`multi_turn/pipeline.py`) and
`evaluate_conversations` (`multi_turn/evaluate.py`). Both are exported from
`redact`. They are not recipe stages: call them from Python.

## Purpose

Build multi-turn conversation datasets for any use case. A **`Setting`**
says who takes part and when the conversation ends; playing it on a seed
gives a **`Trajectory`**, whose step log records every turn and every note.
A separate judge pass scores the saved conversations.

```python
from redact import evaluate_conversations, generate_conversations
from redact.multi_turn import ModelActor, ScriptedActor, Setting

def make_setting():
    return Setting(
        participants=[ScriptedActor("user", ["Go on."]),
                      ModelActor("bot", "venice-uncensored")],
        max_turns=4,
    )

generate_conversations(seeds, make_setting, data_dir="./runs/convo", iterations=2)
evaluate_conversations(data_dir="./runs/convo", judge_model="claude-opus-4-6")
```

## The parts of a conversation

| Name | Module | What it is |
|---|---|---|
| `Setting` | `conversation.py` | `participants` (actors, in turn order), `max_turns` (default 6, the seed included), `stop` (a function of the transcript that ends the conversation early), `name` |
| `Actor` | `actors.py` | An actor's turn is a generator: it yields `LLMRequest`s and returns the steps it adds |
| `ScriptedActor` | `actors.py` | Sends its messages in order, then repeats the last; calls no model |
| `ModelActor` | `actors.py` | One call to its model on the visible transcript, with an optional system prompt |
| `StrategyActor` | `actors.py` | Sends what a `propose(transcript, turn_index)` generator returns; a returned note is logged as a `strategy` step |
| `Transcript`, `Step` | `transcript.py` | The step log. Steps of type `message` and `reply` are sent to models; `strategy`, `analysis` and `evaluation` are notes, only logged |
| `Trajectory` | `conversation.py` | `input_id`, `setting`, `transcript`, `turns_used`, `stop_reason` |

The seed is the first participant's message, turn 0. The other participants
then take turns in order. `run_conversation(setting, seed, call)` plays one
conversation with a blocking call function.

`multiturn_attacks/` builds a `Setting` of an attacker and a target model
(`build_attack_setting`) and runs it through these two functions
(`generate_attacks`, `score_attacks`). `optimization/` searches over
conversation moves with a judge (`optimize`).

## Generating conversations

### Inputs

- `seeds`: a frame with a `seed`, `sample`, `prompt` or `text` column. A
  seed's id is its `sample_id`, else the content id of its text. `category`
  and `entry_type` are copied to the rows when present.
- `setting`: a `Setting`, deep-copied per conversation, or a function that
  returns a new one. Use a function when an actor keeps state.

### Models and settings

The models are those of the `ModelActor`s, by registry name. Settings:
`iterations` (conversations per seed, default 1), `batch_size` (default 256),
`resume`, `output_path`, `resolve` (a `model name -> ModelClient` function,
for tests).

### Unit and ledger key

One unit is one conversation: `(input_id, iteration)`, the setting played
once on one seed.

| | Value |
|---|---|
| Ledger key | `(input_id, iteration)` |
| Ledger record | the key and `stop_reason` |
| Manifest row | `input_id`, `iteration`, `category`, `entry_type`, written again every run |

### Files

`Datasets/conversations.csv`, with `conversations.manifest.jsonl` and
`conversations.state.jsonl` beside it.

| Column | Meaning |
|---|---|
| `sample_id` | Content id of `<input_id>:<iteration>` |
| `input_id` | The seed's id |
| `iteration` | Which conversation of the seed |
| `setting` | The setting's `name` |
| `turns_used` | Turns played, the seed included |
| `stop_reason` | `max_turns`, `stop_condition`, or `ERROR: <message>` |
| `transcript` | The step log as JSON: a list of `{type, actor, content, role, model, meta}` |
| `category`, `entry_type` | Copied from the seed |
| `source` | `conversation` |

### Resume and failures

The stage resumes from its ledger, with the lock and cut-back of
[internals/dataset.md](../internals/dataset.md). Seeds added since the last
run are planned and run. A conversation whose actor raises gets a row with
`stop_reason` `ERROR: <message>`, an empty transcript, and an ack. A failed
call stops the stage.

## Scoring conversations

### Inputs and model

- The conversations: `conversations=`, else `conversations_path`, else
  `Datasets/conversations.csv`.
- `judge_model`, required. Pick a model that is not among those judged.
- `judge_system`: the criterion, as the judge's system prompt. Default: the
  packaged `prompts/conversation/judge/` criterion.
- `scope`: `last_reply` judges the final reply against the seed;
  `transcript` judges the whole conversation.

### Unit and ledger key

One unit is one conversation row, one judge call. There is no plan: the stage
scores rows that exist.

| | Value |
|---|---|
| Ledger key | `sample_id` of the conversation |
| Ledger record | `{"sample_id": ...}` |

### Files

`Datasets/conversations_scored.csv`, beside the conversations file, with
`conversations_scored.state.jsonl`.

| Column | Meaning |
|---|---|
| `sample_id` | The conversation's `sample_id` |
| `success` | Whether the judge's reply reads as a yes |
| `judge_reasoning` | The judge's reply when it is not a yes |
| `judge_model` | The judge |
| `scope` | `last_reply` or `transcript` |

### Resume and failures

Conversations the ledger records are skipped. A conversation too long for the
judge is scored `success` false with `prompt too long: ...` and acked. Any
other error stops the stage.
