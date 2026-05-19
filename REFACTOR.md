# REDACT Refactoring Plan

## Context

REDACT started as an eval-only tool for content moderation, then expanded to jailbreak augmentation, then to training data for a constitutional classifier and jailbreak robustness. The architecture absorbed these additions but shows three seams:

1. **LLM layer**: `batch_generate()` means different things per backend — vLLM does a true single engine pass; API backends loop sequentially. Callers must know which backend they're on. No capability flags. Uncensored vs. standard models mixed without explicit markers.
2. **Checker split**: `build_quality_checker()` is harmful-only. Constitution added `_build_constitution_checker()` in its own module with entry-type awareness rather than extending the existing function. `EntryType` is stranded in `constitution/`.
3. **Scattered config**: Pipeline choices (model, num_turns, entry_types, sampling probabilities) are hardcoded in runner scripts. No unified config schema equivalent to taxonomy.

---

## Architecture — What Uses What

```
configs/
  taxonomy/          ←── dataset/taxonomy.py (loads it)
  pipelines/         ←── dataset/taxonomy.py (new: load_pipeline_config)
  jailbreak/
    combination_spec ←── jailbreak/utils.py (load_spec, lru-cached)
  seeds/             ←── dataset/taxonomy.py (load_seeds)

types.py             ←── content_moderation/checker.py
                     ←── constitution/generation.py
                     ←── dataset/utils.py

llms/
  model_config       ←── llms/api.py (backend routing + get_capabilities)
                     ←── llms/wrappers.py (BatchCaller.from_model)
  base               ←── all backends (venice, anthropic, vllm)
  api (router)       ←── pipelines/* (get_backend)
  wrappers           ←── pipelines/* (BatchCaller, RateLimiter)
  calls              ←── content_moderation/generation.py
                     ←── constitution/input_generation.py
  prompts            ←── content_moderation/generation.py
                     ←── constitution/input_generation.py
  translator         ←── jailbreak/obfuscation/translation.py

dataset/
  io                 ←── content_moderation/generation.py
                     ←── constitution/input_generation.py
                     ←── pipelines/*
  merge              ←── pipelines/dataset.py (build_dataset)
  taxonomy           ←── pipelines/* (iter_categories, load_seeds)
                     ←── pipelines/from_config.py (load_pipeline_config, new)
  utils (new)        ←── pipelines/* (link_constitution_to_inputs, etc.)

content_moderation/
  generation         ←── pipelines/inputs.py
  checker            ←── pipelines/inputs.py
                     ←── constitution/input_generation.py (shared after Step 3)
  metaprompt         ←── pipelines/inputs.py

constitution/
  generation         ←── pipelines/constitution.py
  input_generation   ←── pipelines/constitution_inputs.py

jailbreak/
  techniques/*       ←── jailbreak/utils.py (tag_all_functions)
  utils              ←── jailbreak/engine.py (new: step pool, batch execution)
  engine (new)       ←── pipelines/jailbreaks.py

pipelines/ (new package, replaces pipelines.py)
  from_config        ←── redact/__init__.py
  taxonomy, inputs, constitution, constitution_inputs,
  outputs, jailbreaks, dataset   ←── redact/__init__.py
```

---

## Execution Order

### Phase 0A — Jailbreak techniques (in progress)
*Make each individual technique correct and complete in isolation.*

Each technique function in `obfuscation/`, `hacking/`, `manipulation/`, `requests/` works standalone. All 47+ functions tagged and registered via `tag_all_functions()`. This is the stable primitive layer everything else builds on.

---

### Phase 0B — Jailbreak combination engine (`src/redact/jailbreak/engine.py`, new)
*Locks in the interface. Steps 5 and 7 depend on this being done first.*

**Why this before the rest of the refactor:** The current `combine_techniques()` creates a single chained function per sample and runs it top-to-bottom. For efficient batching you need the inverse: advance all samples through step N together as one LLM batch call, then step N+1, etc. This is a meaningful interface change to `utils.py` — if `pipelines/jailbreaks.py` is written before this is locked in, it would need to be rewritten. Resolving this first means the pipeline layer just calls the engine and never touches retry state or step pools.

**The problem:** Each sample has a different technique combination. Some steps are pure transforms (instant), some are LLM calls (batch-able). A failed step needs to retry with feedback but still produce output tied to the original sample ID. After retry exhaustion the sample is discarded but others continue.

**Design:**

1. **Step pool** — unfold each sample's technique chain into `(sample_id, step_index, technique_fn)` triples. Pure-transform steps are executed immediately; LLM-dependent steps are queued.

2. **Execution engine** (`engine.py`) — advance all samples layer by layer:
   - Collect all samples currently at a pure-transform step → apply instantly
   - Collect all samples at an LLM step → batch call → distribute results back
   - Each sample carries `(current_output, feedback, attempts_remaining)` state

3. **Retry with linkage** — on failure, feedback is attached to the sample's state and the step is retried. Output always links back to the original `input_id`. On exhaustion → `DISCARDED` tag, sample exits the pool.

**New public interface** (what `pipelines/jailbreaks.py` will call):
```python
def batch_apply_combinations(
    samples: list[dict],           # [{"id": ..., "prompt": ..., "combination": fn}, ...]
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    benign_data: pd.DataFrame | None = None,
    num_retries: int = 2,
) -> list[dict]:                   # [{"input_id": ..., "jailbreak": ..., "info": ..., "technique": ...}, ...]
```

`combine_techniques()` in `utils.py` is retained for single-sample use and testing but `engine.py` is the path for pipeline-scale runs.

---

### Step 1 — Shared types (`src/redact/types.py`)
*No dependencies. Pure addition.*

**Why:** `EntryType` lives in `constitution/generation.py`. When Step 3 extends `build_quality_checker()` (in `content_moderation/`) to handle all entry types, it needs to import `EntryType`. If `content_moderation/` imports from `constitution/`, that's a bad dependency direction — a lower-level module depending on a higher-level one. `types.py` at the package root has no dependencies on anything else, so it can be safely imported anywhere.

Create `src/redact/types.py`:
```python
from enum import Enum

class EntryType(str, Enum):
    HARMFUL = "harmful"
    DUAL_USE_HARMFUL = "dual_use_harmful"
    DUAL_USE_BENIGN = "dual_use_benign"
    BENIGN = "benign"

ALL_ENTRY_TYPES: list[EntryType] = list(EntryType)
```

- Update `constitution/generation.py`: replace local definition with `from ..types import EntryType`
- Re-export from `constitution/__init__.py` and `redact/__init__.py` — no caller breakage

---

### Step 2 — LLM layer cleanup

**Why second:** Everything above depends on the LLM layer. Cleaning this up before writing any pipeline files means later steps can call `BatchCaller.from_model()` and `get_model_capabilities()` instead of duplicating backend-detection logic. This step is purely additive — no existing caller signatures change.

**`src/redact/llms/model_config.py`** — add four fields to `ModelConfig`:
```python
is_uncensored: bool = False
supports_system_prompt: bool = True
recommended_max_workers: int = 1
native_batching: bool = False
```

Update registry entries:
| Model | is_uncensored | recommended_max_workers | native_batching |
|---|---|---|---|
| venice-uncensored | True | 3 | False |
| deepseek-v3.2 | True | 2 | False |
| olafangensan-glm-4.7-flash-heretic | True | 2 | False |
| venice-uncensored-vllm | True | 1 | True |
| claude-opus-4-6 | False | 1 | False |

**`src/redact/llms/base.py`** — add property:
```python
@property
def supports_native_batching(self) -> bool:
    return False
```
Override in `VLLMBackend` to return `True`. Callers branch on this instead of `isinstance` checks.

**`src/redact/llms/wrappers.py`** — add `BatchCaller.from_model()`:
```python
@classmethod
def from_model(cls, backend, model, rate_limiter=None) -> "BatchCaller":
    config = get_model_config(model)
    return cls(backend, rate_limiter=rate_limiter,
               max_workers=config.recommended_max_workers)
```
Add guard in `run()`: raise `ValueError` if `backend.supports_native_batching and self.max_workers > 1` — makes the GPU contention footgun an explicit error.

**`src/redact/llms/api.py`** — add `get_model_capabilities(model: str) -> dict` returning the four flags as a plain dict.

---

### Step 3 — Unified checker

**Why here:** Depends on Step 1 (`EntryType` must exist in a neutral place). Done before the pipeline restructure so every pipeline file can call one function regardless of entry type.

**`src/redact/content_moderation/checker.py`** — extend `build_quality_checker()`:
```python
def build_quality_checker(
    category: str,
    criteria: str = "",
    entry_type: str | EntryType = "harmful",
    subcategory: str = "",
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
```

Internal routing:
- `entry_type == "harmful"` → existing path (`content_moderation/quality_check/template.json`)
- All other types → constitution checker path (`constitution/checker/template.json`, injects `entry_type` + `subcategory`)

All existing callers pass no `entry_type` → default `"harmful"` → identical behavior. `_build_constitution_checker()` becomes the internal delegate, no longer part of the public interface.

---

### Step 4 — Dataset additions

**Why here:** Independent of Steps 1–3 except `add_entry_type()` uses `EntryType` from Step 1. Done before pipeline restructure so new pipeline files can import these utilities directly.

Create `src/redact/dataset/utils.py`:
```python
def link_constitution_to_inputs(constitution_df, inputs_df) -> pd.DataFrame:
    # Join on sample_description → end-to-end entry→prompt traceability

def add_entry_type(df, entry_type_col="entry_type") -> pd.DataFrame:
    # Fill missing entry_type with "harmful" for old CSVs (backward compat)

def filter_by_entry_type(df, entry_types: list[str]) -> pd.DataFrame:
    # Thin wrapper around filter_dataset on entry_type column

def technique_version_tag(df, spec_version: str) -> pd.DataFrame:
    # Add combination_spec_version column for reproducibility auditing

def summarize_dataset(df) -> dict:
    # Returns: total, per-category, per-entry_type, accepted rate, source distribution
```

Add `"version": "1.0"` key to `src/redact/configs/jailbreak/combination_spec.json`.

Export all five from `dataset/__init__.py`.

---

### Step 5 — Jailbreak config externalization
*Depends on Phase 0B (engine interface locked) and Step 4 (version tag).*

**Why here:** The sampling probabilities are the last hardcoded generation-control values. Moving them to `combination_spec.json` means the generation config (Step 6) can reference and override them without touching Python code.

**`src/redact/configs/jailbreak/combination_spec.json`** — add:
```json
"sampling_probs": {
  "hacking": 0.5,
  "manipulation": 0.33,
  "requests": 0.7,
  "request_obfuscation": 0.5
}
```

**`src/redact/jailbreak/utils.py`** — in `sample_combination()`:
- Read `spec["sampling_probs"]` from `load_spec()` at function start
- Replace the four hardcoded probability literals with spec values
- Add `sampling_probs: dict | None = None` parameter to override per-call

**`src/redact/pipelines.py`** — add `sampling_probs: dict | None = None` to `generate_jailbreaks()`, forwarded to the engine.

---

### Step 6 — Generation config schema

**Why here:** All pieces (shared types, capability flags, unified checker, dataset utilities, externalized probabilities, locked engine interface) must exist before a config schema referencing them makes sense. This step defines the schema and loader; it doesn't change pipeline behavior yet.

Create `src/redact/configs/pipelines/`.

Schema for a pipeline config JSON:
```json
{
  "name": "my_run",
  "pipeline": "content_moderation | constitution | constitution_to_input | jailbreak | outputs",
  "model": {
    "gen_model": "venice-uncensored",
    "check_model": "venice-uncensored"
  },
  "generation": {
    "num_turns": 10,
    "samples_per_request": 5,
    "use_feedback": true,
    "use_prohibited": true,
    "use_metaprompt": true
  },
  "filtering": {
    "entry_types": null,
    "num_categories": null
  },
  "constitution": {
    "num_categories": 10,
    "entry_types": ["harmful", "dual_use_harmful", "dual_use_benign", "benign"],
    "include_standalone_benign": false
  },
  "jailbreak": {
    "max_complexity": 6,
    "max_obfuscations": 2,
    "seed": 42,
    "pure_only": false,
    "include_hacking": true,
    "include_manipulation": true,
    "include_obfuscation": true,
    "include_requests": true,
    "sampling_probs": null
  },
  "io": {
    "taxonomy": "content_moderation_categories",
    "dataset_dir": null,
    "output_dir": null,
    "fresh": false,
    "save": true
  }
}
```

Add `load_pipeline_config(name, config_dir=None) -> dict` to `src/redact/dataset/taxonomy.py` following the exact pattern of `load_taxonomy()`.

Ship `src/redact/configs/pipelines/full_pipeline.json` as a reference example.

---

### Step 7 — Pipelines restructure
*Depends on all prior steps. Final integration.*

**Why last:** Pulls each pipeline function into its own file (independently navigable and testable) and adds `run_pipeline_from_config()` as the single dispatch entry point. All public names stay the same — nothing outside the package breaks.

```
src/redact/pipelines/
    __init__.py              — re-exports all 7 existing public names + run_pipeline_from_config
    taxonomy.py              — create_taxonomy()
    constitution.py          — generate_constitution()
    constitution_inputs.py   — generate_inputs_from_constitution()
    inputs.py                — generate_inputs()
    outputs.py               — generate_outputs()
    jailbreaks.py            — generate_jailbreaks() — calls engine.batch_apply_combinations()
    dataset.py               — build_dataset()
    from_config.py           — run_pipeline_from_config(config: dict | str) -> pd.DataFrame
```

`from_config.py` dispatches on `config["pipeline"]`, loads JSON by name if string passed, merges sections into kwargs, calls the appropriate pipeline function.

`src/redact/__init__.py` import stays the same (`from .pipelines import ...`). Delete old `pipelines.py`.

---

## Summary Table

| Phase / Step | What | Depends on |
|---|---|---|
| **0A** | Individual jailbreak techniques | — |
| **0B** | Combination engine + batch execution | 0A |
| **1** | Shared `types.py` + `EntryType` | — |
| **2** | LLM layer capability flags + `BatchCaller.from_model` | — |
| **3** | Unified checker with entry_type routing | 1 |
| **4** | Dataset utility functions | 1 |
| **5** | Jailbreak config externalization | 0B, 4 |
| **6** | Generation config schema + loader | 1–5 |
| **7** | Pipelines restructure + `run_pipeline_from_config` | 0B, 1–6 |

Steps 1, 2, and 4 have no inter-dependencies and can be done in parallel.

---

## Critical Files

| File | Phases/Steps |
|---|---|
| `src/redact/jailbreak/engine.py` (new) | 0B |
| `src/redact/jailbreak/utils.py` | 0B, 5 |
| `src/redact/types.py` (new) | 1 |
| `src/redact/constitution/generation.py` | 1 |
| `src/redact/llms/model_config.py` | 2 |
| `src/redact/llms/base.py` | 2 |
| `src/redact/llms/vllm_backend.py` | 2 |
| `src/redact/llms/wrappers.py` | 2 |
| `src/redact/llms/api.py` | 2 |
| `src/redact/content_moderation/checker.py` | 3 |
| `src/redact/constitution/input_generation.py` | 3 |
| `src/redact/dataset/utils.py` (new) | 4 |
| `src/redact/dataset/__init__.py` | 4, 6 |
| `src/redact/configs/jailbreak/combination_spec.json` | 4, 5 |
| `src/redact/pipelines.py` → `src/redact/pipelines/` | 6, 7 |
| `src/redact/dataset/taxonomy.py` | 6 |
| `src/redact/configs/pipelines/full_pipeline.json` (new) | 6 |

---

## Verification

- **0A**: Each technique function produces expected output for a known input; `is_noop` returns False for all non-trivial transforms
- **0B**: `batch_apply_combinations()` with a mixed batch (pure + LLM techniques) produces N outputs linked to original input IDs; DISCARDED samples exit cleanly without blocking others
- **Steps 1–3**: Existing test suite stays green (all public names preserved, defaults unchanged)
- **Step 2**: `get_model_capabilities("claude-opus-4-6")` returns expected flags; `BatchCaller.from_model()` respects `recommended_max_workers`; vLLM + `max_workers>1` raises `ValueError`
- **Step 3**: `build_quality_checker("Physical Harm", entry_type="benign")` produces a checker prompt with benign semantics
- **Step 5**: Generating jailbreaks with `sampling_probs={"hacking": 0.9}` shifts technique distribution measurably
- **Step 6**: `load_pipeline_config("full_pipeline")` parses without error
- **Step 7**: `from redact import generate_jailbreaks, run_pipeline_from_config` both resolve; `run_pipeline_from_config("full_pipeline")` dispatches correctly
