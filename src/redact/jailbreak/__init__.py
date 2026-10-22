"""Jailbreak augmentation: a pool of techniques, a plan of one combination per
input and iteration, and the stage that runs the units.

The techniques live in one subpackage per layer of the spec: ``rewrite/``,
``hacking/``, ``manipulation/``, ``obfuscation/`` and
``requests/``. A transform is
``(prompt) -> (text, info)``; a technique that calls a model is a generator
(:mod:`redact.jailbreak.technique`).

Usage::

    from redact.jailbreak import apply_combination, combine_techniques
    from redact.jailbreak.obfuscation.encoding import to_base64, to_rot13

    combined = combine_techniques(to_rot13, to_base64)
    text, info, accepted, reasoning = apply_combination(combined, prompt)
"""

# One input through its techniques
from .chain import apply_combination, combine_techniques, in_layer_order, run_chain

# The stage: plan, drive the units, save
from .pipeline import run_augmentation

# The plan of a run
from .plan import (
    assign_combination,
    combination_of,
    coverage,
    coverage_path,
    default_escalation_schedule,
    plan_run,
)

# The techniques a run may draw from
from .pool import all_techniques, registry, technique_pool, uses_translation

# Which techniques may still join a combination, and drawing one
from .rules import compatible
from .sampling import sample_combination, sample_exact_combination

# The spec and its tags
from .technique import load_spec, tag

# Every registered technique carries its spec tags from import on.
tag(all_techniques())

__all__ = [
    "all_techniques",
    "apply_combination",
    "assign_combination",
    "combination_of",
    "combine_techniques",
    "compatible",
    "coverage",
    "coverage_path",
    "default_escalation_schedule",
    "in_layer_order",
    "load_spec",
    "plan_run",
    "registry",
    "run_augmentation",
    "run_chain",
    "sample_combination",
    "sample_exact_combination",
    "tag",
    "technique_pool",
    "uses_translation",
]
