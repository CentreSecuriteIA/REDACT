"""Plan whether a run's local models fit on this machine.

Works out each local model's footprint, places the models on the cards, and
produces a plan with a readable explanation, before anything is loaded.

The plan describes what a run does:

- A plan covers models that are resident together; those that cannot be
  do not fit. :func:`plan_phases` makes one plan per stage, for a run that
  unloads between stages what no later stage calls.
- No device is chosen at load time, so a single-GPU vLLM engine always loads
  on card 0 and a tensor-parallel one on cards 0..n-1.
- A vLLM engine is granted its padded need plus a share of its card's spare
  memory, and loads with that grant as its ``gpu_memory_utilization``.

Footprints are estimates, and all sizes are GiB. The plan is a warning ahead
of the run, not a guarantee against running out of memory.
"""

#TODO(driver script): what residency per stage (plan_phases, run_pipeline)
# does not do yet:
#   - Apply the planned card assignment (ModelFootprint.devices) when an
#     engine loads. Until then single-GPU vLLM engines all land on card 0.
#   - Confirm on a GPU that unload_local() returns a vLLM engine's memory.
#   - Replicas of one model across cards for throughput.
#   - Join or cancel the preload thread at exit.
#   - Paraphrase: plan its check model and the engines already resident.
#   - Free memory per card and CUDA_VISIBLE_DEVICES; mixed card sizes.
#   - Bill local cost on the union of engine lifetimes.
#   - A plan does not see engines already loaded, by this process or another.
#     The runner should own this; one process (container) per run is a candidate.

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
