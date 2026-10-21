"""Benign roles: a plain, harmless role given to the assistant before the
prompt, as real users open a request: a helper who knows the subject, a tutor
or coach, or an assistant for a named everyday task.

Each draws one of the twelve variants of its pool in
``requests/benign_role_templates.json``; the prompt is kept as it is. The
roles are ordinary by design: none asks the assistant to set anything aside
or says how to reply. They apply to benign and dual-use benign inputs only
(:data:`ENTRY_TYPES`), as hard negatives, a role in front of a harmless
request, so the plan draws them for no other input
(:func:`~redact.jailbreak.technique.applicable_when`). The natural kind draws
them, the attacks never do (their spec ``kinds``).

``info`` is ``role=<directive>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from redact.types import EntryType

from ..directives import apply_template_directive, read_pools, uses_pool
from ..technique import applicable_when

_FILE = "requests/benign_role_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="role")

#: The entry types a benign role is drawn for.
ENTRY_TYPES = (EntryType.BENIGN, EntryType.DUAL_USE_BENIGN)


def is_benign_input(text: str, entry_type: str) -> bool:
    """Whether a benign role may be drawn for an input: its entry type is one
    of :data:`ENTRY_TYPES`, whatever its text."""
    return entry_type in ENTRY_TYPES


@applicable_when(is_benign_input)
@uses_pool(_FILE, "subject_helper")
def to_benign_role_subject_helper(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Give the assistant the role of a helper who knows the subject."""
    return _apply("subject_helper", prompt, rng=rng)


@applicable_when(is_benign_input)
@uses_pool(_FILE, "tutor")
def to_benign_role_tutor(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Give the assistant the role of a tutor or a coach."""
    return _apply("tutor", prompt, rng=rng)


@applicable_when(is_benign_input)
@uses_pool(_FILE, "task_assistant")
def to_benign_role_task_assistant(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Give the assistant the role of a helper with a named everyday task."""
    return _apply("task_assistant", prompt, rng=rng)


def get_benign_role_functions() -> list[Callable]:
    """Return all benign role technique functions."""
    return [
        to_benign_role_subject_helper,
        to_benign_role_tutor,
        to_benign_role_task_assistant,
    ]
