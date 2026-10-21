"""Impersonation techniques: the prompt from a stated identity: a
trustworthy professional, a position of authority (spec 1.3) or a developer
of the model (spec 1.3).

``info`` is ``type=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/impersonation_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


@uses_pool(_FILE, "good_person")
def to_impersonate_good_person(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Prepend a trustworthy professional identity claim to the prompt."""
    return _apply("good_person", prompt, rng=rng)


@uses_pool(_FILE, "authority")
def to_impersonate_authority(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Prepend a claimed position of authority to the prompt."""
    return _apply("authority", prompt, rng=rng)


@uses_pool(_FILE, "developer")
def to_impersonate_developer(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Prepend a claim to be one of the model's developers to the prompt."""
    return _apply("developer", prompt, rng=rng)


def get_impersonation_functions() -> list[Callable]:
    """Return all impersonation technique functions."""
    return [
        to_impersonate_good_person,
        to_impersonate_authority,
        to_impersonate_developer,
    ]
