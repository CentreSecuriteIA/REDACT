"""Fallacy-failure technique (Zhou 2407.00869): the prompt reshaped as a
request for a deliberately fallacious, flawed or incorrect rendering of the
task, a frame a model may comply with while leaking the real procedure.

``to_fallacy_failure`` draws one of the variants of its pool in
``requests/fallacy_templates.json`` and sets the prompt into it; it is a pure
transform (no model call). It is a request-layer family, so one request per
combination (the layer cap), and its dimension is ``request_shape``, as the
other request frames: it reshapes the finished request.

``info`` is ``directive=fallacy_failure;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/fallacy_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="directive")


@uses_pool(_FILE, "fallacy_failure")
def to_fallacy_failure(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Frame the prompt as a request for a deliberately fallacious answer."""
    return _apply("fallacy_failure", prompt, rng=rng)


def get_fallacy_functions() -> list[Callable]:
    """Return the fallacy-failure technique function."""
    return [to_fallacy_failure]
