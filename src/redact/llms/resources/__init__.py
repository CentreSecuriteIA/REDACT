"""Local GPU footprint estimation and residency planning. All sizes are GiB.

- :mod:`estimate`: what a model will need, computed from its HF config
  without loading it. Planning uses this.
- :mod:`residency`: packs the estimated footprints onto the available GPUs
  and explains the result.
- :mod:`measure`: what a load took, measured afterwards for telemetry.

``residency`` is not imported here because that would create an import cycle:
the local backends import ``resources.measure``, and ``residency`` imports
the backends. Import it explicitly::

    from redact.llms.resources import residency
"""

from . import estimate, measure
from .estimate import estimate_kv_gib, estimate_weights_gib, planning_gib
