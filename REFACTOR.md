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

Deferred (future work): Theme 3 (paraphrase as extra rows + indicator, inputs & outputs,
constitution vs eval) and Theme 4 (multistep jailbreak outputs). Also a follow-up: thread
`prompt_dir` into checker prompts (currently only the top-level generation prompts).
