"""Jailbreak dataset generation pipeline.

Technique modules are organized by attack family:
- obfuscation/ — text transformation attacks (encoding, translation, etc.)
- hacking/ — cognitive/psychological prompt hacking
- manipulation/ — few-shot hacking (FSH) and attention shifting (DAP)

All LLM-dependent functions accept (backend, model, rate_limiter) as params.
Pure transforms (encoding, structural, suffixes, ascii_art) need no LLM.

Usage:
    from redact.jailbreak import combine_techniques
    from redact.jailbreak.obfuscation import get_type_to_getter
    from redact.jailbreak.obfuscation.encoding import to_base64
"""

# Obfuscation registry
from .obfuscation import (
    get_type_to_getter,
    get_obfuscation_types,
    get_all_obfuscation_functions,
)

# Hacking registry
from .hacking import get_hacking_type_to_getter, get_hacking_types

# Manipulation registry
from .manipulation import get_manipulation_type_to_getter, get_manipulation_types

# Utilities
from .utils import combine_techniques

# Distribution
from .distribution import deterministic_balanced_assign, split_by_technique_functions
