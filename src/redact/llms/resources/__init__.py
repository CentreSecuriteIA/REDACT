"""Local GPU footprint estimation and residency planning. All sizes are GiB.

- :mod:`estimate`: what a model will need, computed from its HF config
  without loading it. Planning uses this.
- :mod:`residency`: places the footprints on the machine's cards and
  explains the result.
- :mod:`measure`: the machine's cards, and what an in-process (transformers)
  load took, for telemetry.

``residency`` is not imported here because that would create an import cycle:
the local backends import this package, and ``residency`` imports the
backends. Import it explicitly::

    from redact.llms.resources import residency
"""

from . import estimate, measure
from .estimate import (
    Estimate,
    estimate_kv_gib,
    estimate_weights_gib,
    planning_gib,
)
