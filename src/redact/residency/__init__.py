"""Plan whether a run's local models fit on this machine.

Works out each local model's footprint, places the models on the cards, and
produces a plan with a readable explanation, before anything is loaded.

The plan describes what a run does:

- A plan covers models that are resident together; those that cannot be
  do not fit. :func:`plan_phases` makes one plan per stage, for a run that
  unloads between stages what no later stage calls. An engine that stays
  resident keeps the cards it loaded on. One kept for a later stage is
  unloaded for the stages that do not fit with it, and loaded again.
- A vLLM engine loads on the cards the plan chose: a single-GPU engine on
  the emptiest card with room for it, a tensor-parallel one on untouched
  cards. When the models do not all fit like that, the engines are placed
  on other cards on which they do, if there are any. Cards are
  numbered over those ``CUDA_VISIBLE_DEVICES`` leaves visible, in PCI bus
  order, and each is planned at its own size. A transformers model goes
  where its ``device_map`` puts it.
- Cards with room take copies of the plan's first vLLM engine (vLLM's
  ``data_parallel_size``), unless a count is given for it. A copy is placed
  like the engine: one of a single-GPU engine shares its card, and the
  engine loads with the smallest share any of its cards grants it; one of a
  tensor-parallel engine takes ``min_gpus`` untouched cards. No copy goes to
  a card smaller than the engine's unless a count is given. Copies of a
  tensor-parallel engine are planned, not yet loaded end to end on a GPU.
- A vLLM engine is granted its padded need plus a share of its card's spare
  memory, and loads with that grant as its ``gpu_memory_utilization``.

Footprints are estimates, and all sizes are GiB. The plan is a warning ahead
of the run, not a guarantee against running out of memory.
"""

#TODO(driver script): what residency per stage (plan_phases, run_pipeline)
# does not do yet:
#   - Paraphraser pool: teach plan_phases that a pool's models run one at a
#     time, so the paraphrase stage needs no plan of its own. Its plan covers
#     only the pool and ignores what other stages hold. Only matters with
#     more than one paraphraser, and none is trained yet, so later.
#   - Used memory: the plan sizes each card by its total and assumes it
#     empty; run_pipeline stops a run when another process holds a planned
#     card (driver._check_cards). A plan asked for outside run_pipeline is
#     not checked.
#   - Unloading for a stage is tried with every candidate unloaded at once.
#     With more engines than the search places exactly (_SEARCH_LIMIT),
#     that attempt can miss, and a run one unload would fit stays unfit.

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
