"""Request-structure jailbreak techniques.

Eight technique families that modify *how* a request is posed:
- format: output-format and response-conditioning directives (pure)
- answer_language: ask for the response in a specific language (pure)
- continuation: ask the model to continue interrupted content (pure)
- indirect: embed harmful request inside a legitimate task (pure)
- distractor: add noise before/after the prompt (pure)
- impersonation: claim a trustworthy professional identity (pure)
- temporal: reframe as a past/historical event (pure)
- asking: innocuous framing and detail-maximising directives (pure)

All functions: str -> tuple[str, str]. No LLM dependency.
"""

from .answer import get_answer_functions
from .answer_language import get_answer_language_functions
from .continuation import get_continuation_functions
from .indirect import get_indirect_functions
from .distractor import get_distractor_functions
from .impersonation import get_impersonation_functions
from .temporal import get_temporal_functions
from .asking import get_asking_functions


def get_request_types() -> list[str]:
    """Return list of request technique type names."""
    return list(get_request_type_to_getter().keys())


def get_request_type_to_getter() -> dict:
    """Return mapping of request type name to its getter function."""
    return {
        "answer_format": get_answer_functions,
        "answer_language": get_answer_language_functions,
        "continuation": get_continuation_functions,
        "indirect": get_indirect_functions,
        "distractor": get_distractor_functions,
        "impersonation": get_impersonation_functions,
        "temporal": get_temporal_functions,
        "asking": get_asking_functions,
    }


def get_all_request_functions() -> list:
    """Return flat list of all request technique functions."""
    funcs = []
    for getter in get_request_type_to_getter().values():
        funcs.extend(getter())
    return funcs
