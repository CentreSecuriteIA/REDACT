# API Reference

These pages are generated from the docstrings in `src/redact/`. They cover
the names a user of the library calls; the pages under Internals explain
how the layers fit together.

The stage functions, the recipe runner and the conversation functions are
re-exported from the top-level package:

```python
from redact import (
    generate_constitution, generate_inputs, generate_outputs,
    generate_paraphrases, generate_jailbreaks, build_dataset,
    run_pipeline, load_recipe, load_params,
    generate_conversations, evaluate_conversations,
)
```

::: redact
    options:
      members: false
      show_root_heading: false
      show_source: false

## Packages

- [Stages](stages.md): the stage functions (`generate_*`, `build_dataset`),
  each the public front of one stage.
- [Recipes and runs](runs.md): `run_pipeline`, `load_recipe` and
  `load_params`.
- [LLMs](llms.md): `ModelClient`, `register_model` and the setup classes a
  registry entry is built from.
- [Dataset](dataset.md): rows on disk, the ledger and manifest, resume, and
  the per-category tree.
- [Jailbreak](jailbreak.md): the technique pool, combinations, the plan, and
  the augmentation core.
- [Multi-turn](multi_turn.md): settings, actors and conversation datasets.
- [Residency](residency.md): `plan_residency`, the GPU planner.
- [Telemetry](telemetry.md): the run trace and the cost roll-up.
