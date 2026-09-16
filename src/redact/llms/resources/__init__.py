"""Local GPU resource estimation and residency planning.

Three modules, deliberately separate because they answer different questions:

- :mod:`estimate` — *what will this need?* Computed from the checkpoint's HF
  config, so it works before any load, on any machine, with nothing cached.
  This is what planning runs on.
- :mod:`residency` — *does it fit, and in what order?* Packs estimated
  footprints onto the available devices and explains the result.
- :mod:`measure` — *what did it actually take?* A post-hoc diagnostic for
  telemetry. Never feeds back into planning; see its docstring for why that
  loop was removed.

``residency`` is deliberately **not** imported here. ``backends/vllm.py`` and
``backends/introspection.py`` import ``resources.measure``, and ``residency``
imports ``..backends`` — so binding it eagerly in this ``__init__`` would make
that a circular import at package-load time. ``estimate`` and ``measure`` are
leaves and depend on nothing in the library, so they are safe to re-export.
Import the planner explicitly::

    from redact.llms.resources import residency
"""

from . import estimate, measure
from .estimate import estimate_kv_gib, estimate_weights_gib, planning_gib
