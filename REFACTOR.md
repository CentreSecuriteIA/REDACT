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

---

# Theme 1 + 2 — Interface Uniformity & Folder-as-kwargs (implementation log)

Follow-on restructuring after the Phase 0/Steps 1–7 work above. Goal: make the public
interface uniform and fold folder discovery into a single `data_dir` root. Design docs:
[`.claude/theme1_2_unified_interface_plan.md`](.claude/theme1_2_unified_interface_plan.md),
[`.claude/theme1_interface_comparison.md`](.claude/theme1_interface_comparison.md).

**Locked decisions:** single `data_dir` root · models by role (drop public `backend`) ·
single `resume` (drop `fresh`) · drop dead `base_url` · unified `generate_inputs` · split config
(recipe + input-params) with per-stage run-manifests · eval = no constitution, training = with
constitution · multi-round jailbreak escalation preserved.

## Stage 1a — single-source path module (mechanical, no behavior change)

**New:** `src/redact/paths.py` — one place that derives every default file/dir from a working
root (`get_output_dir()` unless a root is passed). Layout constants (`DATASETS_DIRNAME`,
`DATA_CACHE_DIRNAME`, `SAMPLES_FILENAME`, …) + per-artifact resolvers: `datasets()`,
`data_cache()`, `category_csv()`, `jailbreaks_csv()`, `output_responses_csv()`,
`complete_dataset_csv()`, `benign_csv()`, `constitution_dir()`, `constitution_inputs_dir()`,
`taxonomy_dir()`.

**Repointed to `paths` (killed duplicated computations):**
- `dataset/io.py` — `_default_dataset_dir()` → `paths.datasets()`; `_resolve_path()` →
  `paths.category_csv()`; `samples.csv` literals → `paths.SAMPLES_FILENAME`.
- `dataset/merge.py` — `samples.csv` sentinels → `paths.SAMPLES_FILENAME`.
- `pipelines.py` — `_default_dataset_dir` / `_default_jailbreak_path` / `_default_benign_path`
  delegate to `paths.*`; **deleted dead `_default_scenario_dir`**; `_DEFAULT_TAXONOMY_DIR` →
  `paths.taxonomy_dir()`; `output_responses.csv` / `complete_dataset.csv` / inline per-category
  `samples.csv` → `paths.*`.
- `jailbreak/manipulation/benign.py` — `_default_benign_path()` → `paths.benign_csv()`.
- `constitution/generation.py` + `constitution/input_generation.py` — inline
  `Data_cache/constitution` and `Datasets/constitution_inputs` → `paths.constitution_dir()` /
  `paths.constitution_inputs_dir()`.

**Eliminated:** two `_default_dataset_dir`, two `_default_benign_path`, two
`Data_cache/constitution` computations, and the dead `_default_scenario_dir`.

**Behavior:** unchanged — all names delegate to identical results; public signatures untouched
(that's Stage 1b). Circular-import safe (`paths` imports only `get_output_dir`, same pattern
`io.py` already used).

**Verification:** import smoke-test resolves all paths; full suite **548 passed, 9 skipped**.

## Stage 1b — `data_dir` root on the public API (additive)

Introduced a single working root, `data_dir`, on every public generation/build function.
All default artifact paths now derive from it via `paths.*(data_dir)`. Kept the existing
`dataset_dir` / `output_dir` / `output_path` params working as **explicit overrides** (they win
over the `data_dir`-derived default); Stage 3 removes/renames the now-redundant ones. When
`data_dir` is `None`, every path is byte-identical to before — fully backward compatible.

**Signatures (added params):**
- `generate_constitution(…, data_dir, output_dir, taxonomy_dir, …)` — const CSVs →
  `paths.constitution_dir(data_dir)` unless `output_dir` given; taxonomy loaded via `taxonomy_dir`.
- `generate_inputs(…, data_dir, dataset_dir, taxonomy_dir, prompt_dir, …)` — **standalone** inputs →
  `paths.datasets(data_dir)`; **constitution-seeded** inputs **auto-route** to
  `paths.constitution_inputs_dir(data_dir)` (no more manually passing `dataset_dir=CONSTITUTION_INPUTS_DIR`).
- `generate_outputs(…, data_dir, dataset_dir, output_path, …)` — reads inputs from
  `paths.datasets(data_dir)`; responses CSV → `paths.output_responses_csv(data_dir)`.
- `generate_jailbreaks(…, data_dir, output_path, benign_path, …)` — **new** `data_dir` +
  `benign_path`; now reads inputs from `paths.datasets(data_dir)` (previously an un-redirectable
  `merge_all()`); jailbreaks CSV → `paths.jailbreaks_csv(data_dir)`; benign cache →
  `paths.benign_csv(data_dir)`.
- `build_dataset(…, data_dir, dataset_dir, …)` — inputs/jailbreaks/merged all derive from `data_dir`.

**Threaded:** `taxonomy_dir` → `load_taxonomy(config_dir=…)`; `prompt_dir` → the top-level
`load_prompt(...)` calls in `generate_inputs`. (Checker-prompt `prompt_dir` threading is a
documented follow-up — checker prompts still use the bundled default.)

**Bug fixed:** `ConstitutionInputPipeline.__init__` now resolves its dirs first and constructs the
inner `InputPipeline(dataset_dir=self.output_dir)` up front, so per-category CSVs land in the
configured dir instead of defaulting to `Datasets/`. Removed the run-time
`self.input_pipeline.dataset_dir = self.output_dir` patch.

**Verification:** `paths.*(root)` relocate correctly; `ConstitutionInputPipeline(output_dir=X)` →
inner `dataset_dir == X`; full suite **548 passed, 9 skipped**.

## Stage 2 — models by role

Every generation function now selects its model **by role**, resolves the backend from the model
name, and no longer exposes a `backend` param on the public API.

- **Dropped public `backend`** from `generate_inputs` / `generate_outputs` / `generate_jailbreaks` /
  `generate_constitution`. Backend is always `get_backend(model)` (internal `_get_backend` helper
  retained for tests + advanced use). No notebook passed `backend=`; one README example updated in
  Stage 6.
- **Role-default `model`** — signatures changed `model: str = "<literal>"` → `model: str | None =
  None`; each body resolves `model or default_model_for_role(<role>)`: inputs/outputs/jailbreaks →
  `uncensored_gen` (venice-uncensored), constitution → `constitution_gen` (claude-opus-4-6). The
  effective default is unchanged; it's just role-driven now, so re-registering a role's model
  changes every default at once.
- **`check_model`** unchanged — plain model param, `None` falls back to `model`.
- **`translation_model`** added to `generate_jailbreaks`. Threaded through
  `batch_apply_combinations(translate_model=…)` → `make_combination_gen` → the translation technique
  (which already accepted `translate_model`, defaulting to the `translation` role = DeepSeek).
  Translation still routes to its own model independently of `model`; `include_translation=False`
  still drops the family entirely.

**Verification:** role lookups resolve as expected; `backend` gone + `model` defaults `None` on all
four; a driven translation generator emits its `LLMRequest` against the passed `translation_model`;
full suite **548 passed, 9 skipped**.

## Stage 3 — resume-only + cleanup

Collapsed the redundant/dead params so the public surface matches the design.

- **Single `resume`** everywhere; deleted `fresh`. `resume=False` now means "clear + restart":
  `generate_inputs` (standalone clears category CSVs / constitution mode passes `fresh=not resume`),
  `generate_outputs` (clears CSV + ledger). `generate_jailbreaks` / `generate_constitution` already
  had `resume`.
- **Deleted dead `base_url`** from `generate_inputs` / `generate_outputs` / `generate_jailbreaks`
  (the real endpoint lives in `venice_backend.py`; the params were never wired).
- **`chunk_size` → `batch_size`** in `generate_jailbreaks` (one batch-size name across the library).
- **Deleted deprecated `generate_inputs_from_constitution`** + its `__init__` export
  (`generate_inputs(constitution_df=…)` is the path). `ConstitutionInputPipeline` kept (still tested).
- **Removed redundant directory params** now covered by `data_dir`: `dataset_dir` from
  `generate_inputs` / `generate_outputs` / `build_dataset`, `output_dir` from `generate_constitution`.
  Single-file `*_path` overrides (`output_path`, `inputs_path`, `responses_path`, `jailbreak_path`,
  `benign_path`, `manifest_path`) are retained as power-user knobs.
- **`create_taxonomy`**: `config_dir` → `taxonomy_dir`, added `verbose` (print now gated).

**Tests touched:** `tests/test_pipelines.py` — `create_taxonomy(config_dir=…)` → `taxonomy_dir=…`
(4 sites). Everything else already used kept params (io/merge/class `dataset_dir`/`output_dir`,
`generate_jailbreaks(output_path=…)`).

**Verification:** signature assertions confirm the removed params are gone and the canonical ones
present; deprecated fn no longer importable; full suite **548 passed, 9 skipped**.

## Stage 4 — config layer

Config-driven runs: a run is now a **recipe** + **input-params** JSON, and every stage leaves a
manifest the next stage can read.

**New `src/redact/runconfig.py`:**
- `write_manifest` / `read_manifest` / `manifest_path` — per-stage `{stage}.run.json` under
  `{data_dir}/Datasets/`, recording resolved params, models, row counts, spec version, timestamp.
- `load_recipe` — parse + validate (dataset_type ∈ {eval, training}; known stages; eval can't
  include the `constitution` stage); records the recipe's `_base_dir` for relative `params_file`.
- `load_params` — load the input-params file (resolved relative to the recipe dir).
- `run_pipeline(recipe, params=None)` — the driver. Maps recipe + params → the `generate_*`
  kwargs, runs the named stages in order, threads the constitution DataFrame into `generate_inputs`
  for `training` runs, and writes a manifest per stage. Returns a summary dict.
- Exported from `redact`: `run_pipeline`, `load_recipe`, `load_params`, `write_manifest`,
  `read_manifest`.

**Config split (per the locked decision):** `augmentations` (jailbreak `include_*`/`pure_only`)
live in the **recipe** (structural); numeric tunables live in the **input-params** file. `eval` vs
`training` = presence of the `constitution` stage / `dataset_type`.

**New files:** `configs/runs/{eval_example,training_example,params_example}.json`;
`scripts/run.py` (recipe → pipeline, `--params`/`--data-dir`/`--quiet`); `scripts/inspect_run.py`
(summarize a run's manifests).

**Tests:** `tests/test_runconfig.py` (9) — manifest round-trip, recipe/params validation, and an
**offline** `run_pipeline` over jailbreaks+build (pure_only, no network) asserting counts +
manifests + cross-stage chaining; plus a training-without-constitution-stage error. Also verified
via the CLI end-to-end (`run.py` → manifests → `inspect_run.py`).

**Verification:** full suite **557 passed, 9 skipped**; packaged example recipe + `params_file`
resolve; CLI run produced jailbreaks (3 rows) + build (6 rows) with both manifests.

## Stage 5 — notebooks reorg

All notebooks moved into **`notebooks/`** and rewritten on the new interface (single `data_dir`,
role-defaulted models, `resume`, no `base_url`/`chunk_size`/`dataset_dir`). Framed around the
eval-vs-training split:

- **`notebooks/eval_pipeline.ipynb`** — no constitution: standalone inputs → outputs → jailbreaks →
  build eval dataset. Ends with a one-shot `run_pipeline({...eval...})` cell.
- **`notebooks/training_pipeline.ipynb`** — with constitution: constitution → seeded inputs →
  outputs → jailbreaks → build training dataset. Ends with a one-shot `run_pipeline({...training...})`.
- **`notebooks/content_moderation.ipynb`** — per-stage inputs+outputs, plus a `create_taxonomy`
  (`taxonomy_dir`) + custom-taxonomy demo.
- **`notebooks/jailbreak_augmentation.ipynb`** — per-stage jailbreaks (pure-only toggle, escalation
  schedule) + build.

Old root notebooks (`full_pipeline`, `constitution_generation`, `content_moderation`,
`jailbreak_augmentation`) removed. Verified: all four parse as valid nbformat-4; no notebook
references any removed param.

## Stage 6 — tests + docs

- **Tests** were kept green stage-by-stage (config-driven changes updated `tests/test_pipelines.py`
  in Stage 3; `tests/test_runconfig.py` added in Stage 4). Final: **557 passed, 9 skipped**.
- **README.md** — Quick Start rewritten around `data_dir` + role-defaulted models + `run_pipeline`;
  replaced the removed `generate_inputs_from_constitution` example and the `generate_inputs(backend=…)`
  example (now `register_model(...)` → auto-resolve); `fresh` → `resume` in the constitution section.
- **CLAUDE.md** — architecture tree gains `paths.py`, `runconfig.py`, `notebooks/`, `scripts/`,
  `configs/runs/`; Usage section rewritten to the new surface (one `data_dir`, models by role, no
  public `backend`, `run_pipeline`, retained `*_path`/`taxonomy_dir`/`prompt_dir` overrides).
- Packaging: hatchling includes `configs/runs/*.json` as package data by default (same as existing
  `configs/*.json`); no `pyproject.toml` change needed.

---

## Refactor complete — new public interface at a glance

| Concept | Before | After |
|---|---|---|
| Output location | `dataset_dir` / `output_dir` / `output_path` (5 names) | one **`data_dir`** root + optional single-file `*_path` overrides |
| Path resolution | duplicated `_default_*` in 5 modules | single-source **`redact/paths.py`** |
| Model / backend | `model` literal defaults + public `backend` | **`model=None` → role default**; backend auto-resolved (no public `backend`) |
| Translation model | implicit only | explicit **`translation_model`** on `generate_jailbreaks` |
| Restart idiom | `fresh` / `resume` / both / neither | single **`resume`** everywhere |
| Batch size | `batch_size` vs `chunk_size` | **`batch_size`** everywhere |
| Dead params | `base_url` (×3) | removed |
| Deprecated fn | `generate_inputs_from_constitution` | removed (use `generate_inputs(constitution_df=…)`) |
| Config / orchestration | hardcoded in notebooks | **recipe + input-params JSON + per-stage manifests**; `run_pipeline` / `scripts/run.py` |
| Notebooks | 4 at repo root, stale API | `notebooks/` — eval (no constitution) vs training (with) + 2 per-stage |
| Constitution inputs | manual `dataset_dir=…` | auto-route to `Datasets/constitution_inputs/` (+ propagation bug fixed) |

Deferred (future work): Theme 3 (paraphrase — see below) and Theme 4 (multistep jailbreak
outputs). Also a follow-up: thread `prompt_dir` into checker prompts (currently only the
top-level generation prompts).

---

# Theme 3 — Paraphrase / Fingerprint Removal (in progress)

Design/plan: `.claude/theme3_paraphrase_plan.md` (v5). Adds paraphrased copies of base inputs
and accepted base outputs as additive artifacts (like `jailbreaks.csv`), K 1:1 calls
round-robin over a new `paraphraser` role, a separate meaning-preservation checker
(check→drop), merged per `build_dataset(mode=…)` (eval → separate `paraphrased.csv`, training
→ merged). No folder reorg, no `kind` column (artifact file + metadata field carry provenance).

## T3.0 — paraphraser role + real `paraphrase_batch` (done)

- `paraphrase.py`: replaced the identity stub with a real **batched** `paraphrase_batch`
  (dispatch via `BatchCaller`, prompt from `paraphrase/template.json`) + `paraphrase_sample`
  wrapper (with `system_prompt` override). Updated `tests/content_moderation/test_paraphrase.py`
  (was asserting pass-through) → asserts real model output + call shape.
- `model_config.py`: new **`paraphraser`** role.

**Decision logged:** added a **dedicated `"venice-paraphraser"` model with its own identity**
(not an alias / not repurposing a gen model). As a stand-in until the external defingerprinting
model is registered, it **loads the same HF weights** as `venice-uncensored-vllm`
(`hf_model_id="dphn/Dolphin-Mistral-24B-Venice-Edition"`, `backend_type="vllm"`) — its own model
that happens to load the same weights. Swap `hf_model_id` / `register_model(role="paraphraser")`
for the real one when it lands. GLM stays `uncensored_gen`. Because it's a distinct model, the
paraphrase checker default (`uncensored_gen` = venice-uncensored) is now genuinely independent
of the paraphraser. Verified: `paraphraser → venice-paraphraser` (vLLM, Dolphin weights), full
suite **567 passed, 9 skipped**.

**Future note (no code, deferred):** optionally prepend a short random token (e.g. `osdjoi27 `)
to the text before paraphrasing to push each attempt in a different direction — also reusable in
finetuning for in-context variety. Not implemented.

## T3.1 — meaning-preservation checker (done)

New `prompts/content_moderation/paraphrase_check/template.json` + `checker.build_paraphrase_checker`
(payload `"ORIGINAL:\n…\n\nPARAPHRASE:\n…"` via `paraphrase_check_payload`), parsed by the existing
`batch_check_samples` (starts-with-"yes" → accept, else drop). Exported from `content_moderation`.
Test: `tests/content_moderation/test_paraphrase_checker.py`.

## T3.2 / T3.3 — `generate_paraphrases` (done)

`pipelines.generate_paraphrases(inputs, outputs, data_dir, paraphraser, check_model,
paraphrases_per_sample, target, check, resume, batch_size, seed, …)`:
- Paraphrases **base inputs** and/or **accepted base outputs** (target `inputs`/`outputs`/`both`).
- K units per sample (K repeated 1:1 calls), paraphraser assigned **round-robin** (hash of
  `seed:base_id:k`; trivial with 1 model). Warns on K>1 with a single paraphraser.
- **Model-grouped batched execute**, paraphrase via `paraphrase_batch`, **separate** meaning
  checker via `batch_check_samples` (check→drop). Warns if checker == paraphraser (placeholder).
- **Dedup**: drops no-op (== original) and duplicate paraphrase texts.
- Writes `paraphrases_inputs.csv` / `paraphrases_outputs.csv` (id, `input_id`=base id, `iteration`,
  sample, category, entry_type, `paraphrase_model`, accepted, reasoning, source). Per artifact:
  a `*.manifest.jsonl` = the full plan (every unit → its model), and a sidecar `*.state.jsonl`
  **ledger** (mirrors constitution / output-response) recording every *attempted* unit +
  model + status (`accepted`/`rejected`/`dropped_dedup`). **Resume reads the ledger**, so drops
  are never silently re-attempted, and different paraphrasers are mapped per unit.
- New `paths.py` helpers: `paraphrases_inputs_csv` / `paraphrases_outputs_csv` / `paraphrased_csv`.
- Exported `generate_paraphrases`. Tests: `tests/test_paraphrase_pipeline.py` (dedup, resume,
  check-drop, accepted-only outputs — all offline via monkeypatch).

**Resolved (was: "dropped units re-attempt on resume").** Originally resume was driven by the
output CSV, so units that produced no row (dedup no-ops) were re-attempted on a re-run. **Now
switched to a sidecar `*.state.jsonl` ledger** (the pattern the user pointed to from constitution
generation / jailbreak augmentation): every *attempted* `(base_id, iteration)` unit is appended
with its `paraphrase_model` + `status` after the chunk's CSV write, and resume reads the ledger —
so nothing is ever silently re-attempted, and the plan/ledger cleanly map units to (possibly
different) paraphrasers. The `check_model==paraphraser` guard-warning remains but no longer fires
by default (dedicated `venice-paraphraser` ≠ venice-uncensored checker). Test:
`test_ledger_prevents_reattempt_of_dropped_units`.

## T3.4 — `build_dataset` paraphrase merge (done)

`build_dataset(mode="training"|"eval", include_paraphrases, paraphrase_*_path)`: **training** merges
accepted paraphrases into `complete_dataset.csv` (`dataset_type="content_moderation_paraphrase"`);
**eval** keeps them out and writes a separate `paraphrased.csv`. Base always present; jailbreaks
still only ever built from base inputs. Compatible with the existing auto-detect (no folder reorg).

## T3.5 — `paraphrase` stage in runconfig (done)

`STAGES` gains `paraphrase`; `run_pipeline` dispatches it (`models.paraphraser` /
`models.paraphrase_check`, `params.paraphrase`), and the `build` stage now passes
`mode=dataset_type` (+ `params.build`, e.g. `include_paraphrases`). Recipe surface lets a run select
the paraphraser(s) and control the merge. Test: `test_run_pipeline_paraphrase_stage`.

## T3.6 — docs + finishing pieces (done)

- CLAUDE.md (arch line + Usage), README paraphrase blurb + **placeholder disclaimer**, example
  configs (paraphrase stage + params.paraphrase + build.include_paraphrases).
- **`prompt_dir` threaded like the other stages** (`None` → bundled `prompts/`): added to
  `generate_paraphrases` (→ paraphrase prompt **and** paraphrase-check prompt) and to
  `generate_outputs` (→ gen prompt **and** output checker); `run_pipeline` passes the recipe
  `prompt_dir` into the outputs + paraphrase stages. Test: `test_prompt_dir_threads_to_paraphrase_and_check`.
- **Notebook:** `notebooks/content_moderation.ipynb` gained a separate "Paraphrase" section
  (own cells) with the ⚠️ disclaimer that the real paraphraser is not used (placeholder loads
  Dolphin-24B weights).

**Status:** Theme 3 code-complete + offline-tested — full suite **569 passed, 9 skipped**. The
only remaining item is a **live smoke-test** (below), which needs real LLM access.

---

## T3 — Live smoke-test runbook (TODO — for whoever runs it live)

Everything above is validated with **mocked LLMs** only. Do a one-off live run to confirm the
paraphrases are real, meaning-preserving, and that check→drop / dedup / ledger-resume behave.

### Prerequisites
- `.env` with `VENICE_API_KEY` (drives `uncensored_gen` gen/check models). `pip install -e .`.
- A **paraphraser**. The default `paraphraser` role resolves to `venice-paraphraser`, a **vLLM**
  placeholder (loads `dphn/Dolphin-Mistral-24B-Venice-Edition`) — that needs `pip install -e ".[vllm]"`
  + a GPU. **To smoke-test without a GPU**, pass an API paraphraser explicitly, e.g.
  `paraphraser="olafangensan-glm-4.7-flash-heretic"` (an uncensored Venice model — distinct from the
  default `check_model` venice-uncensored, so validation stays independent).

### Steps (small, ~a minute of API calls)
```python
from redact import generate_inputs, generate_outputs, generate_paraphrases, build_dataset
D = "./runs/smoke_paraphrase"

inputs  = generate_inputs(data_dir=D, samples_per_category=3, num_categories=1)
outputs = generate_outputs(data_dir=D, inputs=inputs, max_per_category=3)

para = generate_paraphrases(
    data_dir=D, inputs=inputs, outputs=outputs, target="both",
    paraphraser="olafangensan-glm-4.7-flash-heretic",   # or None -> vLLM placeholder (GPU)
    paraphrases_per_sample=2,                            # exercises the K>1 warning + dedup
)
print({k: len(v) for k, v in para.items()})

build_dataset(data_dir=D, mode="training")   # merges paraphrases into complete_dataset.csv
build_dataset(data_dir=D, mode="eval")       # writes a separate paraphrased.csv instead
```
Then re-run `generate_paraphrases(... , resume=True)` unchanged to check resume.

### What to verify (checklist)
1. **Real rewording:** `Datasets/paraphrases_inputs.csv` `sample` differs from the base prompt
   (not identical), similar length, **meaning preserved**, and **harmful stays harmful** (not softened).
2. **Separate checker works:** some rows `accepted=False` with a `reasoning` when meaning drifted;
   `build_dataset` (accepted-only for paraphrase) excludes those.
3. **Dedup:** no two identical `sample` values; with `paraphrases_per_sample=2` a single paraphraser
   emits the K>1 UserWarning and near-duplicate K's collapse.
4. **Ledger/resume:** `paraphrases_inputs.state.jsonl` has one line per attempted `(input_id,iteration)`
   with `paraphrase_model` + `status`; `paraphrases_inputs.manifest.jsonl` has the full plan; the
   `resume=True` re-run makes **no new LLM calls** and adds no rows.
5. **Outputs are accepted-only:** `paraphrases_outputs.csv` only paraphrases base outputs that were
   `accepted=True` (refusals skipped).
6. **Merge modes:** training → paraphrase rows present in `complete_dataset.csv`
   (`dataset_type=="content_moderation_paraphrase"`); eval → absent there but present in `paraphrased.csv`.
7. **Config path (optional):** run via `scripts/run.py` with a recipe whose `stages` include
   `paraphrase`; confirm `paraphrase.run.json` manifest + the artifacts.

### Wiring the REAL defingerprinting model
Register it and it takes over the role automatically:
```python
from redact.llms import register_model
register_model("my-defingerprinter", rpm=..., backend_type="venice"|"vllm",
               hf_model_id="...",  # if vllm
               role="paraphraser")
```
or edit the `venice-paraphraser` entry's `hf_model_id` in `llms/model_config.py`.

---

# Theme 4 — Multi-turn conversation datasets (in progress)

Plan/design: `.claude/theme4_multiturn_plan.md` (general `multi_turn/` core → jailbreak
`multiturn_attacks/` + `optimization/` as apps; Inspect-oriented; settings→trajectories with a
typed step log; optional/separable judge). Theme 5 (agentic/eval-setting *data*) shares `Setting`.

## MT.0 / MT.1 — conversation primitives + `multi_turn/` core (done, START)

- **`llms/conversation.py`** (new, model-layer): moved `LLMRequest` here (the routing type);
  added `Step` (typed: strategy/message/reply/analysis/evaluation), `Transcript` (step log +
  `as_messages()` rendering only the visible turns; provenance events never sent to models), and
  `drive_sync` (drive one generator with a blocking call fn). `jailbreak/protocol.py` now
  **re-exports** `LLMRequest` from here (same class → engine untouched; verified by a test).
- **`multi_turn/`** (new package): `Actor` protocol (`turn(transcript) -> Generator[LLMRequest,
  str, list[Step]]`), `ScriptedActor` (seed/replay, no LLM), `ModelActor` (one LLM turn); `Setting`
  (drive mode + participants + max_turns + optional stop predicate); `Trajectory`;
  `conversation_gen` / `run_conversation` (single-conversation runner: seed = turn 0, then
  alternate, logging every step). Wired into `redact.__init__`.
- **Tests:** `tests/test_multi_turn.py` (5) — shared `LLMRequest` class, transcript renders only
  visible steps, 2-actor alternate conversation, stop condition, provenance-not-rendered. Full
  suite **574 passed, 9 skipped**.

**Scope note (deliberate):** this is the single-conversation (sync) core only. Still to come
(MT.later): the batched `drive_generators` extraction from `jailbreak/engine` (+ engine refactor
to use it), the `generate_conversations` pipeline (plan manifest + state ledger + resume + artifact),
the separable `evaluate_conversations` judge layer, and the apps — `multiturn_attacks/`
(jailbreak-composed attacks) and `optimization/` (branch/backtrack search; W1 = a search controller
above the driver). The jailbreak engine was intentionally **not** touched yet.

## MT.2 — batched `drive_generators` extraction (done)

Extracted the round-by-round, pool-by-model driver from `jailbreak/engine` into
`llms/conversation.drive_generators(gens, *, router, finalize, on_error, verbose, progress)` — the
generic engine behind both the jailbreak combination engine and the conversation runner.
`jailbreak/engine.batch_apply_combinations` now builds its per-sample generators and calls
`drive_generators` (behavior identical; the only change is a cosmetic progress-label prefix, one
test updated). Pure refactor — full suite stayed green.

## MT.3 — `generate_conversations` pipeline (done)

`multi_turn/pipeline.generate_conversations(seeds, setting, data_dir, variants_per_seed, resume,
batch_size, …)`: plays a `Setting` (a template deep-copied per unit, or a factory) on each seed ×
variant, driven at scale by `drive_generators`. Full-plan `*.manifest.jsonl` + sidecar
`*.state.jsonl` **ledger** + resume on `(seed_id, variant)` (same pattern as paraphrase/outputs).
Writes `conversations.csv` (`id, seed_id, variant, setting, turns_used, stop_reason, transcript`
[JSON step log], `category`, `entry_type`, `source`). New `paths.conversations_csv`. Exported
`generate_conversations` from `redact`. Tests in `tests/test_multi_turn.py` (offline fake router:
conversation content, resume, variants). Full suite **576 passed, 9 skipped**.

**Docs:** README ("Multi-turn Conversations" subsection) + CLAUDE.md (architecture tree + `llms/`
`conversation.py` note + a `multi_turn/` module section) updated for Theme 4.

**Still to build:** MT.4 separable `evaluate_conversations` (optional judge, per-turn or end);
MT.5 `multiturn_attacks/` (own folder — `StrategyActor` composing jailbreak augmentations +
crescendo, required target, success judge); MT.6 `optimization/` (own folder — branch/backtrack
search controller above the driver, W1).

## MT.4 — separable `evaluate_conversations` (done)

`multi_turn/evaluate.evaluate_conversations(conversations|data_dir, judge_model, judge_system,
scope="last_reply"|"transcript", resume, batch_size, …)`: an **optional, separate** pass over
existing trajectories (generate → then evaluate). Extracts the seed goal + final reply (or renders
the transcript) from the stored step log, judges via `batch_check_samples` with a **user-supplied
criterion** (judge model chosen by the caller, independent of judged models), and writes
`conversations_scored.csv` (`id, success, judge_reasoning, judge_model, scope`), resumable via a
sidecar ledger keyed on conversation id. Exported `evaluate_conversations`. Offline test (mocked
judge → mixed success + resume). Full suite **577 passed, 9 skipped**.

**General `multi_turn/` subsystem status: COMPLETE** — `Setting → Trajectory` generation
(`generate_conversations`) + separable evaluation (`evaluate_conversations`), reusing `llms/`
(`drive_generators`, `Transcript`), Inspect-shaped. Remaining are the **applications in their own
folders**: MT.5 `multiturn_attacks/` (jailbreak-composed multi-turn attacks) and MT.6
`optimization/` (branch/backtrack search controller, W1).

## MT.5 — `multiturn_attacks/` (own folder, done)

Multi-turn jailbreak attacks built on `multi_turn/` (never inside `jailbreak/`). Added a general
`StrategyActor` to the core (adaptive multi-step turn; stateless — `turn_index` derived from the
transcript, so factory/deepcopy-safe). `multiturn_attacks/`:
- `attacks.py` — `crescendo_propose` (template escalation, optional pure-transform jailbreak
  augmentation per turn) and `pair_propose` (attacker-LLM rewrites each next turn from transcript+goal).
- `setting.py` — `build_attack_setting(target_model, attack, …) -> Setting factory` (attacker
  StrategyActor + **required** safety-trained target ModelActor); `SUCCESS_JUDGE_SYSTEM`.
- `__init__` — `generate_attacks` (= build setting → `generate_conversations`) and `score_attacks`
  (= `evaluate_conversations` with the success judge). The architecture pays off: an attack is just
  a Setting fed to the general pipeline + the success judge.
- Tests (`tests/test_multiturn_attacks.py`, 5): crescendo logs strategy + escalates; jailbreak
  technique (`to_rot13`) applied per turn; PAIR uses the attacker model; required-target guard; scoring.

## MT.6 — `optimization/` (own folder, done)

The W1 **search controller above `drive_generators`**: `optimize(seed, target_model, candidates,
judge_model, judge_system, beam, depth, stop_on_success)` runs a **beam/tree search over
conversation moves** — expands every frontier node's candidates in parallel (batched by model),
scores each with the judge, logs the whole `tree` (list of `Node`s), keeps the best `beam`,
continues, early-stops on success. Objective-general (candidates + judge define the objective).
Subsumes PAIR (beam=1), TAP (beam>1), optimized-crescendo (candidates = technique-combinations).
`multiturn_attacks/optimize.py` bridges it: `jailbreak_candidates` (plain follow-up + one variant
per jailbreak technique) + `optimize_attack`. Tests (`tests/test_optimization.py`, 3): finds
success + early stop, expands to depth without success, jailbreak-technique bridge.

**Theme 4 COMPLETE (code):** general `multi_turn/` (conversations + separable evaluation) +
`multiturn_attacks/` (attacks) + `optimization/` (search) — three folders, correct layering
(`multi_turn/ → llms/`; apps → `multi_turn/`). Full suite **585 passed, 9 skipped**. Remaining is
polish: a `conversations`/`attacks` stage in `runconfig`, a notebook, docs for the apps, and a live
smoke-test (all mocked-LLM offline so far). MVP notes: optimization candidates are concrete moves
(attacker-LLM-proposed candidates + per-node checkpoint resume are noted extensions).

## Test coverage pass (thorough)

Added comprehensive offline test coverage for **everything built this session** — **100% line
coverage** on all new modules (measured with `coverage`): `paths.py`, `llms/conversation.py`,
`multi_turn/{core,pipeline,evaluate}.py`, `multiturn_attacks/*`, `optimization/*`, `runconfig.py`,
`content_moderation/paraphrase.py` (687 statements, 0 missed).

New/extended test files (all offline, mock-based — `FakeRouter` for `drive_generators`/pipelines,
`MockBackend` for judges/checkers, monkeypatched `generate_*` for `run_pipeline` stages):
- `tests/test_paths.py` — every path resolver (root default vs explicit, category_csv, taxonomy_dir).
- `tests/test_conversation_primitives.py` — `drive_generators` multi-round batching, immediate
  completion, error isolation (`on_error` vs re-raise), batch-failure isolation, `drive_sync`,
  verbose progress label.
- `tests/test_multi_turn.py` — actor abstract/empty/system-prompt, `conversation_gen` guard,
  pipeline (missing text col, empty seeds, id column, actor-error isolation, deepcopy template,
  verbose+fresh, ledger bad-line robustness), evaluate (last_reply/transcript scopes, validation,
  bad-JSON, resume, fresh).
- `tests/test_multiturn_attacks.py` — crescendo strategy+escalation, jailbreak-technique
  composition, PAIR attacker-model use, required-target/unknown-attack guards, empty-goal.
- `tests/test_optimization.py` — success + early-stop, expand-to-depth, jailbreak bridge,
  required-target, empty candidates, batch-failure isolation.
- `tests/test_runconfig.py` — all-stages `run_pipeline` (mocked generators), params-file loading,
  `write_manifest` extra/spec, `_counts(None)`.

Also fixed a real robustness bug the tests surfaced: `evaluate_conversations` now creates its
output directory (failed when handed a DataFrame directly with no existing `Datasets/`).

Full suite: **622 passed, 9 skipped.**

## Stage 6 — resume unification (one shared ledger) + multi-turn naming harmonization

Every batched stage now resumes through **one** implementation, and the multi-turn
subsystem uses the library's own vocabulary instead of invented terms.

**Shared ledger (`dataset/ledger.py`, new).** `Ledger(path, key_fields, casters)` — the
single sidecar-`*.state.jsonl` implementation: `.completed()` (scalar key for one field,
tuple for several), `.record(rows)` (extra status fields written verbatim, ignored on
read), `.reset()`, `.sidecar(artifact, …)`. Bad-line-robust, mkdir-on-write, composite-key
+ caster aware. Exported as `redact.dataset.Ledger`. Tests: `tests/test_ledger.py` (12).

**Migrated the 5 existing hand-rolled ledgers onto it** (behavior-preserving; on-disk
field names unchanged so committed runs still resume): constitution (`unit`), CM-output
(`input_id`), paraphrase (`(input_id, iteration)`) — kept as thin adapters; conversations
and evaluate migrated directly. The per-module `_read_state`/`_append_state` copies are
gone (logic lives once in `Ledger`).

**Closed the two coverage gaps** the audit found:
- **CM-input-from-constitution** now writes `constitution_inputs.state.jsonl` keyed
  `(sample_description, entry_type, style)`, acked per entry after its rows are saved;
  resume = ledger ∪ existing-CSV (back-compat); `fresh` clears both (style-aware fresh
  keeps other styles' acks). Tests: `tests/content_moderation/test_constitution_input_ledger.py`.
- **Jailbreaks** now write `jailbreaks.state.jsonl` keyed `(input_id, iteration)`, acked per
  chunk after the CSV append; resume = ledger ∪ output-CSV; `resume=False` clears both. The
  `*.manifest.jsonl` plan stays. Tests added to `tests/jailbreak/test_pipeline_jailbreaks.py`.

Intentional exception (documented, not a gap): **standalone meta-prompt input** generation
has no discrete idempotent units (open-ended toward a per-category sample count) — it resumes
by extending its CSVs, same convention as before.

**Multi-turn naming harmonization** (the cluster was drifting from library vocabulary; all
uncommitted, so renamed freely — no on-disk back-compat needed):
- `seed_id` → **`input_id`** (Trajectory field, `conversation_gen`/`run_conversation` param,
  `conversations.csv` column, ledger key). Matches jailbreak/output/paraphrase's source-row
  reference, so conversation datasets join to base inputs uniformly.
- `variant` → **`iteration`** (column + ledger key); param `variants_per_seed` → **`iterations`**.
  Unit key is now `(input_id, iteration)` — identical to paraphrase and jailbreaks.
- optimization search: `Node.reason` / tree-record `reason` → **`judge_reasoning`** (matches
  `evaluate_conversations`). Kept `success` (correct scorer term, already consistent across
  evaluate + optimization). Updated the affected tests.

**Docs:** README gained a "Resume model (one ledger everywhere)" subsection (per-stage unit-key
table); CLAUDE.md gained a `dataset/ledger.py` tree entry + module bullet, a "Resume model" Key
Design Decision, updated jailbreak / CM-input / constitution resume notes, and an updated
`multi_turn/` section documenting the library-consistent naming.

Full suite: **639 passed, 9 skipped.**

## Stage 7 — manifest unification + modular sidecar system + `run_category` deprecation

Did for **manifests** what Stage 6 did for ledgers, and factored the two into one
modular foundation so the resume/plan subsystem is a single system, not two copies.

**Modular foundation (`dataset/sidecar.py`, new).** `JsonlSidecar` owns the shared
sidecar-JSONL mechanics — `.sidecar(artifact)` path resolution, `.exists()`/`.reset()`,
mkdir-on-write, and bad-line-robust `_iter_records()`/`_write(rows, mode)`. Both concrete
types now subclass it:
- `Ledger(JsonlSidecar)` (`dataset/ledger.py`) — refactored onto the base; **public API and
  on-disk format unchanged** (all Stage-6 tests + call sites green). Append-only, keyed,
  `.completed()`.
- `Manifest(JsonlSidecar)` (`dataset/manifest.py`, new) — the plan half: `.write(rows)`
  (idempotent overwrite) / `.load()`. Exported from `dataset/__init__.py` alongside `Ledger`
  and `JsonlSidecar`. Tests: `tests/dataset/test_manifest.py` (11).

`_write` now uses `ensure_ascii=False` (readable unicode in every sidecar file; reads
unaffected) — preserves the jailbreak manifest's prior bytes exactly.

**Migrated the 3 existing hand-rolled manifests onto `Manifest`** (behaviour-preserving,
same on-disk format): `jailbreak/manifest.py` (`plan_run`/`load_plan`; also fixed its stale
"output CSV is the source of truth" docstring — Stage 6 added the jailbreak ledger), the
inline paraphrase manifest (`pipelines.py: _run_paraphrase_target`), and the conversations
manifest (`multi_turn/pipeline.py`).

**Added manifests to the 3 stages that lacked one** (full rollout; each keyed identically to
that stage's existing ledger, written before the batch loop):
- Constitution generation → `constitution.manifest.jsonl` (`(source_category, entry_type)` +
  standalone-benign).
- CM output (`generate_outputs`) → `output_responses.manifest.jsonl` (`input_id`).
- CM input from constitution (`run_from_constitution`) → `constitution_inputs.manifest.jsonl`
  (`(sample_description, entry_type, style)`).

New manifest-presence tests: `tests/constitution/test_generation.py`, `tests/test_pipelines.py`
(incl. the first offline `generate_outputs` run + a "manifest is the full plan even on a partial
resume" test), `tests/content_moderation/test_constitution_input_ledger.py`. Conversations already
assert their manifest (`tests/test_multi_turn.py`), jailbreak via `load_plan`. Coverage of
`dataset/{sidecar,ledger,manifest}.py` is 100% (82 statements, 0 missed).

**`run_category` deprecation.** Standalone meta-prompt input generation is deprecated in favour
of constitution-seeded generation (better/more adjustable coverage) but **kept fully
functional**: `InputPipeline.run_category` and the standalone branch of `generate_inputs`
(`constitution_df is None`) each emit a `DeprecationWarning` + docstring note. Tests assert the
warning fires (`pytest.warns`). It remains the one stage with neither manifest nor ledger.

**Docs:** README rewritten to fit — "Resume model" subsection (manifest+ledger table, modular
`JsonlSidecar` foundation), architecture tree (`dataset/{sidecar,ledger,manifest}.py`), the
Dataset-Functions table (`Ledger`/`Manifest`/`JsonlSidecar` rows), the jailbreak execution-model
paragraph (stale "resumable from the output CSV" → ledger ∪ CSV), the paraphrase note, and a
Quick-Start deprecation callout for standalone `generate_inputs`. CLAUDE.md: tree entries
(`sidecar.py`/`manifest.py`), `dataset/` module bullets, the "Resume model" Key Design Decision,
per-stage resume notes (constitution / CM-output / CM-input each gained a manifest line), and the
standalone-deprecation note. `.claude/manifest_unification_plan.md` marked DONE.

Deferred (separate, user-owned): the deep-internals introspection backend
(`.claude/introspection_backend_plan.md`) — not touched here; this work kept its invariants
(unchanged id/column shapes; `BatchCaller` remains the untouched dispatch chokepoint).

## Stage 8 — deep-internals logging backend (`.claude/introspection_backend_plan.md`)

Implements the plan deferred at the end of Stage 7. Adds a fourth `LLMBackend` capability
flag and a new backend for research/interpretability logging (hidden states, attention,
logprobs) — not a vLLM extension, since vLLM's continuous batching + paged KV-cache discard
intermediate activations by design (long-standing upstream limitation, not fixable here).

**`supports_internals` capability flag** (`llms/base.py`), mirroring `supports_native_batching`
/ `supports_parallel_calls`. Default `False`; only `TransformersIntrospectionBackend` overrides
to `True`.

**`TransformersIntrospectionBackend`** (`llms/introspection_backend.py`, new) — raw HF
`transformers` (lazy-imported, optional dependency), not TransformerLens: covers everything
scoped for v1 (`output_hidden_states`/`output_attentions`/`output_scores` on `.generate()`),
works with any `AutoModelForCausalLM`, no extra dependency weight. TransformerLens's
`HookedTransformer` hook-point API is the better tool once the roadmap reaches *acting on*
internals (GCG-style optimization, steering) — left as a future separate backend, not a v1
dependency. `capture` config (`logprobs` / `hidden_states`: `False`/`"last"`/`"all"` /
`attention`) defaults light (logprobs + last-hidden-state) since attention scales
`layers × heads × seq_len²` and this backend has no batching to amortize storage over.
`supports_native_batching=False`, `supports_parallel_calls=False` (single GPU, same reasoning
as `VLLMBackend`), `supports_internals=True`. Registered via `backend_type=
"transformers_introspect"` + `hf_model_id` + a new `ModelConfig.introspect_kwargs` field
(mirrors `vllm_kwargs`); `api.get_backend()` gained a matching branch, cached by
`f"introspect:{model}"`.

**Identity: no signature threading, one guard in `BatchCaller`.** `internals_id` (single) /
`internals_ids` (batch) are explicit params only the introspection backend and `BatchCaller`
know about — `calls.py`/`router.py` needed zero changes, since `**kwargs` already flows through
them untouched (confirmed still true after Stage 7). `BatchCaller.run()`/`.batch_generate()`
(`wrappers.py`) check `backend.supports_internals` before dispatch and raise `ValueError`
immediately if unsupported — never a silent crash inside Venice/Anthropic's SDK client call
three layers down. When supported, `run()` unzips `internals_ids[i]` into each sequential call;
the (non-existent, since this backend doesn't have one) native-batch path would forward the
whole list — implemented generically so any future native-batching + internals-capable backend
gets it for free.

**Wired into `generate_outputs()`** (`pipelines.py`) as the concrete drop-in proof point: when
the resolved backend's `supports_internals` is `True`, each row gets
`internals_id = _hash_text(f"{input_id}:generate")` — reusing the exact `input_id` the ledger
already keys on, not a new id scheme — passed as `internals_ids` into the existing
`BatchCaller.batch_generate()` call and written as a literal `internals_id` CSV column, so any
row round-trips to `{log_dir}/{internals_id}/` with no separate bookkeeping. Column only appears
when the backend supports it — zero behavior change for the default Venice/vLLM path. Checker-
side capture intentionally left out of v1.

Tests: `tests/llms/test_introspection_backend.py` (registry/routing/capability-flag, offline —
model load itself needs GPU + `transformers`, not exercised here), `tests/llms/test_wrappers.py`
(`TestBatchCallerInternalsIds` — guard raises / no-op when unsupported / per-item unzip / native-
batch forwarding / length-mismatch raises), `tests/test_pipelines.py`
(`TestGenerateOutputsInternalsId` — column absent/present/matches-hash/reaches-backend).

Docs: CLAUDE.md gained the `introspection_backend.py` tree entry, a `supports_internals` backend
bullet, and a Key Design Decision explaining the id-reuse + single-guard approach.

Full suite: **672 passed, 10 skipped** (1 new skip: the one test needing `transformers`/`torch`,
not installed on this machine).

## Stage 9 — `sample_id` library-wide fix + folder-per-`input_id` internals capture

Design review of Stage 8 (in conversation, not code) surfaced that the `internals_id` CSV
column violated the codebase's own principle for this feature — internals logging must be a
**non-interfering hook** (never touches a pipeline's CSVs/schema/control-flow/ledger/manifest,
only writes to `{log_dir}/...`). Redesigned around reusing `input_id` directly as the capture
root instead, which surfaced a **separate, independently-justified gap**: not every pipeline's
output rows had their own content-hash identity (`sample_id`, distinct from `input_id`, which
only points back to the *origin* sample).

**`sample_id` audited and fixed everywhere** (own merits — a library data-model fix, not
something internals logging needed): CM input had `id` (renamed → `sample_id`); CM output had
none (added, `_hash_text(output_response)`); jailbreak had a **broken** `id` — inherited from
the input row via `meta_map` in `generate_jailbreaks()`, so every technique/iteration variant of
one input silently shared `input_id`'s value instead of having its own (fixed in
`jailbreak/engine.py: _finalize()`/`_finalize_error()`, now `_hash_text(jailbreak_text)`);
paraphrase and conversations already had it correctly, just named `id` (renamed → `sample_id`
for consistency); constitution-generation entries had none (added via `ConstitutionEntry.
__post_init__`). Touched: `dataset/io.py`, `dataset/merge.py` (`normalize_csv`,
`merge_csvs_from_dirs`, all three `*_KEEP_COLUMNS` presets), `pipelines.py` (all three
pipelines), `jailbreak/manifest.py` (`plan_run`'s `id_col` default), `multi_turn/pipeline.py` +
`multi_turn/evaluate.py`, `constitution/generation.py`. Every test referencing the old column
name updated; new regression tests assert the jailbreak fix specifically (`sample_id` distinct
per technique/iteration, not a duplicate of `input_id`) in both
`tests/jailbreak/test_engine.py` and `tests/jailbreak/test_pipeline_jailbreaks.py`.

**Internals capture rewired onto a folder-per-`input_id` scheme.** The `internals_id` CSV
column is gone. `generate_outputs()` now passes `internals_ids = [f"{input_id}/output", ...]`
(and `.../val_out` for the checker call — wired for both, not just generation) straight to
`BatchCaller`; no backend/wrapper code changes were needed to support this, since
`internals_id` is just a string and `introspection_backend.py`'s existing
`self._log_dir / internals_id` + `mkdir(parents=True)` already nests correctly on the embedded
`/`. `TransformersIntrospectionBackend` now also writes an always-on `meta.json` per capture
(resolved model/settings, input messages, output text) alongside whatever tensors `capture`
asks for — load-bearing, not just convenient: multi-round jailbreak techniques discard
intermediate prompts/completions once a row is finalized, so this is the only place that data
survives at all. Added `llms/wrappers.py: assert_single_sample_per_call(backend,
samples_per_call)`, the shared guard for pipelines shaped like `run_from_constitution` (one
LLM call can return several samples, so a single forward pass's internals can't be attributed
to any one of them) — exported, not yet wired into any pipeline.

**Deliberately deferred** (design-only, captured in `.claude/introspection_backend_plan.md`):
jailbreak/paraphrase/input-generation capture *wiring* itself (independent-iteration and
retry-loop folder nesting); `translation.py`'s retry-loop index threading (only serves the
deferred jailbreak wiring); calling `assert_single_sample_per_call` from any pipeline (no
caller needs it yet).

Docs: `CLAUDE.md` (`sample_id`/`input_id` pairing convention as a new Key Design Decision, the
non-interference principle, the folder-scheme description replacing the old column description,
`assert_single_sample_per_call`), `README.md` (conversation-scoring ledger key).

Full suite: **680 passed, 10 skipped** (same 1 pre-existing skip — needs `transformers`/`torch`,
not installed on this machine).

## Stage 10 — internals capture wired into input/val_in, paraphrase, and jailbreak

Closes out everything Stage 9 deferred (`input`/`val_in`/jailbreak/paraphrase capture *wiring*,
the `samples_per_entry=1` gate actually being called). Every generation pipeline except
multi-turn conversations now supports internals capture end-to-end.

**New mechanism this required, not anticipated in Stage 9: `LLMBackend.rename_capture(old_id,
new_id)`** (default no-op; real move-the-folder implementation in
`TransformersIntrospectionBackend`). Case B's folder (`{input_id}/{stage}/{sample_id}/`) is
keyed by the *resulting* text's own hash, but `internals_id` has to be supplied *before* the
call — paraphrase's and jailbreak's output text isn't known until the call returns. Resolved by
capturing under a pre-call-known **provisional** id (`{bid}/paraphrase/attempt_{k}`;
`{id}/jailbreak/attempt_{iteration}`), then relabeling to the real id once the text is hashed.
One rename moves the whole subtree, so jailbreak's nested per-technique captures move with it
for free.

**`content_moderation/generation.py: run_from_constitution`** — `input` (generation) is gated by
`llms/wrappers.py: assert_single_sample_per_call` (added in Stage 9, called for the first time
here): raises if internals capture is requested with `samples_per_entry != 1`, since one forward
pass can't be attributed to more than one resulting sample. **Correction to the Stage 9 design
doc**: `val_in` (the checker) is *not* gated — it dispatches one call per already-extracted
sample in a flat batch, so it's never at risk of that attribution problem. `input` roots at the
constitution entry's own composite hash (pre-call known, matches the entry's ledger key);
`val_in` roots at the extracted sample's own hash (its eventual `sample_id`, knowable before the
checker call since the text already exists by then).

**`pipelines.py: _run_paraphrase_target`** — provisional-id-then-rename, per model group.

**Jailbreak — the deep one.** `llms/conversation.py: LLMRequest` gained an `internals_id` field;
`drive_generators` collects it per model-per-round batch and forwards to
`router.batch_generate(..., internals_ids=...)`, but only when at least one request in that
batch actually wants it (a list of all-`None` would otherwise still trip `BatchCaller`'s guard
on a non-supporting backend, since only a bare `None` — the kwarg omitted — reads as "no capture
requested"). `jailbreak/utils.py: _run_chain` gained `_tag_yields`, which wraps a technique
generator's yields and tags each with a sequential `internals_id` (`{root}/{technique}/{n}`) —
**checked per request, against that specific request's own target model's backend** (not a
single run-level flag), so a chain mixing an internals-capable `gen_model` with a non-capable
`translate_model` captures exactly what's capturable and never trips the guard on the other.
`jailbreak/engine.py: batch_apply_combinations` builds each sample's provisional root
(`{id}/jailbreak/attempt_{iteration}`) when `capture_internals` is on; `_finalize`/
`_finalize_error` compute the real `sample_id` and call `rename_capture` to relabel the whole
provisional subtree.

**Two corrections to the Stage 9 design doc, both discovered while implementing:**
- Jailbreak's per-technique nesting ended up as **generic sequential numbering**
  (`{technique}/{n}`), not a `_retry` suffix distinguishing retry-loops (translation's own
  translate→check→retry) from genuinely multi-step techniques (cognitive hacking's
  scenario→construction) — that distinction isn't visible at the `_run_chain` level without each
  technique explicitly cooperating, which none do. Every LLM call still gets its own distinct,
  capturable path either way. **Consequence: `translation.py`'s planned retry-index threading
  turned out to be unnecessary** — `_tag_yields`' generic per-yield numbering already covers it,
  untouched.
- Jailbreak's actual folder order is `{input_id}/jailbreak/{sample_id}/{technique}/{n}/`, not the
  design doc's `jailbreak_{technique}/{sample_id}/` (technique before sample_id) — `sample_id`
  has to be the direct parent of the per-technique breakdown so one `rename_capture` call moves
  the whole subtree; technique-first would need a separate rename per technique.

Tests: `tests/content_moderation/test_constitution_input_internals.py`,
`tests/test_paraphrase_internals.py`, `tests/jailbreak/test_engine_internals.py` (incl.
`TestMixedModelSupport`, a regression test for the per-request guard-tripping bug above),
`tests/test_conversation_primitives.py` (`drive_generators`' `internals_ids` forwarding),
`tests/llms/test_introspection_backend.py::TestRenameCapture`.

Docs: `CLAUDE.md` (`introspection_backend.py` bullet and the internals-capture Key Design
Decision both updated for `rename_capture` and full wiring status). `.claude/
introspection_backend_plan.md` updated to Stage 10 status.

Full suite: **702 passed, 10 skipped** (same 1 pre-existing skip).
