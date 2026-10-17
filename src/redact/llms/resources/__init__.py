"""Local GPU footprint estimation and machine measurement. All sizes are GiB.

- :mod:`estimate`: what a model will need, computed from its HF config
  without loading it. :mod:`redact.residency` plans with this.
- :mod:`measure`: the machine's cards, and what an in-process (transformers)
  load took, for telemetry.
"""

from . import estimate, measure
from .estimate import (
    Estimate,
    estimate_kv_gib,
    estimate_weights_gib,
    planning_gib,
)
