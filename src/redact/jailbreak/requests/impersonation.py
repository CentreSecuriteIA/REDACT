"""Impersonation technique: the prompt from a stated professional identity.

``info`` is ``type=good_person;variant=<variant>``.
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


def get_impersonation_functions() -> list[Callable]:
    """Return all impersonation technique functions."""
    return [
        to_impersonate_good_person,
    ]
