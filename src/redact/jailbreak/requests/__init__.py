"""Request techniques, all of them pure transforms."""

from collections.abc import Callable

from .answer import get_answer_functions
from .answer_language import get_answer_language_functions
from .asking import get_asking_functions
from .benign_role import get_benign_role_functions
from .code_framing import get_code_framing_functions
from .continuation import get_continuation_functions
from .distractor import get_distractor_functions
from .fallacy import get_fallacy_functions
from .guardrail_address import get_guardrail_address_functions
from .impersonation import get_impersonation_functions
from .indirect import get_indirect_functions
from .smalltalk import get_smalltalk_functions
from .temporal import get_temporal_functions


def get_all_request_functions() -> list[Callable]:
    """Return flat list of all request technique functions."""
    return (
        get_answer_functions()
        + get_answer_language_functions()
        + get_continuation_functions()
        + get_indirect_functions()
        + get_distractor_functions()
        + get_impersonation_functions()
        + get_temporal_functions()
        + get_asking_functions()
        + get_fallacy_functions()
        + get_code_framing_functions()
        + get_guardrail_address_functions()
        + get_smalltalk_functions()
        + get_benign_role_functions()
    )
