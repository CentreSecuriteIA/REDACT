"""Jailbreak techniques, and the chain, sampler and plan that combine them.

The techniques live in one subpackage per layer of the spec: ``hacking/``,
``manipulation/``, ``obfuscation/`` and ``requests/``. A transform is
``(prompt) -> (text, info)``; a technique that calls a model is a generator
(:mod:`redact.jailbreak.technique`).

Usage::

    from redact.jailbreak import apply_combination, combine_techniques
    from redact.jailbreak.obfuscation.encoding import to_base64
"""

from .chain import apply_combination, combine_techniques, in_layer_order, run_chain
from .engine import batch_apply_combinations

# Run planning
from .plan import (
    assign_combination,
    combination_of,
    default_escalation_schedule,
    plan_run,
)
from .pool import all_techniques, registry, technique_pool, uses_translation

# Request protocol + batched engine
from .protocol import LLMRequest, run_sync
from .rules import compatible
from .sampling import (
    sample_combination,
    sample_exact_combination,
)
from .technique import load_spec, tag

build_combination = combination_of  # the name pipelines.py imports

# Every registered technique carries its spec tags from import on.
tag(all_techniques())
