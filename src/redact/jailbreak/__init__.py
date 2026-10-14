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

# Hacking registry
from .hacking import get_all_hacking_functions

# Manipulation registry
from .manipulation import get_all_manipulation_functions
from .obfuscation import get_all_obfuscation_functions

# Run planning
from .plan import (
    assign_combination,
    build_function_registry,
    combination_of,
    default_escalation_schedule,
    plan_run,
)

# Request protocol + batched engine
from .protocol import LLMRequest, run_sync

# Requests registry
from .requests import get_all_request_functions
from .rules import compatible
from .sampling import (
    sample_combination,
    sample_exact_combination,
)
from .technique import load_spec, tag

build_combination = combination_of  # the name pipelines.py imports

# ---------------------------------------------------------------------------
# Tag all technique functions with compatibility metadata at import time
# ---------------------------------------------------------------------------

tag(
    get_all_obfuscation_functions()
    + get_all_hacking_functions()
    + get_all_manipulation_functions()
    + get_all_request_functions()
)
