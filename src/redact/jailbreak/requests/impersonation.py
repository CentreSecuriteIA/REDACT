"""Impersonation technique: the prompt from a stated professional identity.

``info`` is ``type=good_person;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "impersonation_templates.json")
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


def to_impersonate_good_person(prompt: str) -> tuple[str, str]:
    """Prepend a trustworthy professional identity claim to the prompt."""
    return _apply("good_person", prompt)


def get_impersonation_functions() -> list[Callable]:
    """Return all impersonation technique functions."""
    return [
        to_impersonate_good_person,
    ]
