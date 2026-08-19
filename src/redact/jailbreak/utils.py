"""Backward-compatible re-export shim for jailbreak technique composition.

This module used to hold 7 unrelated concerns in one ~820-line file (spec
loading, chain execution/generator-tagging, compatibility filtering, two
sampling strategies, escalation-schedule config, rejection parsing, and
manifest-assignment/reconstruction). It's now split into focused modules —
this file just re-exports everything under its original name so every
existing ``from redact.jailbreak.utils import X`` / ``from .utils import X``
keeps working unchanged:

- :mod:`redact.jailbreak.spec` — load_spec, tag_all_functions
- :mod:`redact.jailbreak.chain` — combine_techniques, make_combination_gen,
  apply_combination, is_noop, and the private chain-execution helpers
  (_normalize_output, _select_kwargs, _tag_yields, _run_chain,
  _parse_rejection_info)
- :mod:`redact.jailbreak.sampling` — get_compatible_remaining,
  sample_combination, sample_exact_combination, default_escalation_schedule,
  and the private helpers used only by them (_pick_by_family,
  _is_request_obfuscation_compatible)
- :mod:`redact.jailbreak.assignment` — assign_combination, build_combination,
  build_function_registry, and _stable_seed

New code should import directly from the specific submodule above; this file
exists for backward compatibility with existing imports (including tests
that reach into the private ``_``-prefixed names).
"""

from .assignment import (
    _stable_seed,
    assign_combination,
    build_combination,
    build_function_registry,
)
from .chain import (
    _normalize_output,
    _parse_rejection_info,
    _run_chain,
    _select_kwargs,
    _tag_yields,
    apply_combination,
    combine_techniques,
    is_noop,
    make_combination_gen,
)
from .sampling import (
    _is_request_obfuscation_compatible,
    _pick_by_family,
    default_escalation_schedule,
    get_compatible_remaining,
    sample_combination,
    sample_exact_combination,
)
from .spec import load_spec, tag_all_functions

__all__ = [
    "load_spec",
    "tag_all_functions",
    "combine_techniques",
    "make_combination_gen",
    "apply_combination",
    "is_noop",
    "get_compatible_remaining",
    "sample_combination",
    "sample_exact_combination",
    "default_escalation_schedule",
    "assign_combination",
    "build_combination",
    "build_function_registry",
    # Private names kept importable here for existing test coverage
    # (tests/jailbreak/test_utils.py etc.) that reaches into them directly.
    "_normalize_output",
    "_select_kwargs",
    "_tag_yields",
    "_run_chain",
    "_parse_rejection_info",
    "_pick_by_family",
    "_is_request_obfuscation_compatible",
    "_stable_seed",
]
