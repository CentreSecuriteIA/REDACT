"""Jailbreak dataset generation pipeline.

Technique modules are organized by attack family:
- obfuscation/ — text transformation attacks (encoding, translation, etc.)
- hacking/ — cognitive/psychological prompt hacking
- manipulation/ — few-shot hacking (FSH) and attention shifting (DAP)
- requests/ — request-structure attacks (answer format, continuation, indirect)

All LLM-dependent functions accept (backend, model) as params.
Pure transforms (encoding, structural, suffixes, ascii_art, requests) need no LLM.

Usage:
    from redact.jailbreak import combine_techniques, sample_combination
    from redact.jailbreak.obfuscation import get_all_obfuscation_functions
    from redact.jailbreak.obfuscation.encoding import to_base64
"""

from .chain import apply_combination, combine_techniques, is_noop, make_combination_gen
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
