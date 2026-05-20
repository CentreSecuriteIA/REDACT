"""Jailbreak dataset generation pipeline.

Technique modules are organized by attack family:
- obfuscation/ — text transformation attacks (encoding, translation, etc.)
- hacking/ — cognitive/psychological prompt hacking
- manipulation/ — few-shot hacking (FSH) and attention shifting (DAP)
- requests/ — request-structure attacks (answer format, continuation, indirect)

All LLM-dependent functions accept (backend, model, rate_limiter) as params.
Pure transforms (encoding, structural, suffixes, ascii_art, requests) need no LLM.

Usage:
    from redact.jailbreak import combine_techniques, sample_combination
    from redact.jailbreak.obfuscation import get_type_to_getter
    from redact.jailbreak.obfuscation.encoding import to_base64
    from redact.jailbreak.requests import get_request_type_to_getter
"""

# Obfuscation registry
from .obfuscation import (
    get_type_to_getter,
    get_obfuscation_types,
    get_all_obfuscation_functions,
)

# Hacking registry
from .hacking import (
    get_hacking_type_to_getter,
    get_hacking_types,
    get_all_hacking_functions,
)

# Manipulation registry
from .manipulation import (
    get_manipulation_type_to_getter,
    get_manipulation_types,
    get_all_manipulation_functions,
)

# Requests registry
from .requests import get_request_type_to_getter, get_request_types, get_all_request_functions

# Utilities
from .utils import (
    combine_techniques,
    make_combination_gen,
    tag_all_functions,
    get_compatible_remaining,
    sample_combination,
    apply_combination,
    assign_combination,
    build_combination,
    build_function_registry,
    is_noop,
    load_spec,
)

# Request protocol + batched engine
from .protocol import LLMRequest, run_sync
from .engine import batch_apply_combinations

# Run planning + resumable manifest ledger
from .manifest import (
    plan_run,
    load_plan,
    plan_index,
    completed_from_output,
    compute_sample_id,
    default_manifest_path,
)

# Distribution
from .distribution import deterministic_balanced_assign, split_by_technique_functions

# ---------------------------------------------------------------------------
# Tag all technique functions with compatibility metadata at import time
# ---------------------------------------------------------------------------

tag_all_functions(
    get_all_obfuscation_functions()
    + get_all_hacking_functions()
    + get_all_manipulation_functions()
    + get_all_request_functions()
)
