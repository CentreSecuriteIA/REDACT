"""Plan whether a run's local models fit on this machine.

Works out each local model's footprint, places the models on the cards, and
produces a plan with a readable explanation, before anything is loaded.

The plan describes what a run does:

- A plan covers models that are resident together; those that cannot be
  do not fit. :func:`plan_phases` makes one plan per stage, for a run that
  unloads between stages what no later stage calls. An engine that stays
  resident keeps the cards it loaded on.
- A vLLM engine loads on the cards the plan chose: a single-GPU engine on
  the card with the most room, a tensor-parallel one on untouched cards.
  Cards are numbered over those ``CUDA_VISIBLE_DEVICES`` leaves visible. A
  transformers model goes where its ``device_map`` puts it.
- Cards left idle take copies of the plan's first single-GPU vLLM engine,
  one per card (vLLM's ``data_parallel_size``), unless a count is given
  for it.
- A vLLM engine is granted its padded need plus a share of its card's spare
  memory, and loads with that grant as its ``gpu_memory_utilization``.

Footprints are estimates, and all sizes are GiB. The plan is a warning ahead
of the run, not a guarantee against running out of memory.
"""

#TODO(driver script): what residency per stage (plan_phases, run_pipeline)
# does not do yet:
#   - Confirm on a GPU that unload_local() returns a vLLM engine's memory.
#   - Join or cancel the preload thread at exit.
#   - Paraphraser pool: teach plan_phases that a pool's models run one at a
#     time, so the paraphrase stage needs no plan of its own. Its plan covers
#     only the pool and ignores what other stages hold. Only matters with
#     more than one paraphraser, and none is trained yet, so later.
#   - Free memory per card. One card size per plan: the smallest.
#   - A plan does not see engines already loaded, by this process or another.
#     The runner should own this; one process (container) per run is a candidate.
#   - Placement is greedy: models are placed in the order given and never
#     moved, so a set that could fit, mostly across several cards, can be
#     reported as not fitting. Needs a search over arrangements.
#   - An engine stays resident from its first stage to its last, so a stage
#     can fail to fit only because of an engine kept for a later stage.
#     Releasing and reloading it would fit, at the cost of a second load.
#   - Copies: one that shares its card with another model, a fallback to one
#     copy when the build fails, and copies of a tensor-parallel engine
#     without a count given.
#   - A transformers load that initialises CUDA while a vLLM engine is being
#     built sees only that engine's cards (backends/vllm.py: _visible_devices).

import logging

from .explain import report
from .footprint import ModelFootprint, footprint
from .lifecycle import apply_plan, loaded_engines, preload, unload_local
from .phases import Phase, plan_phases
from .placement import plan_residency
from .plan import ResidencyPlan

logger = logging.getLogger(__name__)

__all__ = [
    "ModelFootprint",
    "Phase",
    "ResidencyPlan",
    "apply_plan",
    "footprint",
    "loaded_engines",
    "plan_phases",
    "plan_residency",
    "preload",
    "report",
    "unload_local",
]
