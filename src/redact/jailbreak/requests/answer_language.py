"""Answer-language techniques, one per language of ``configs/jailbreak/languages.json``
marked ``answer_language``.

``info`` is ``language=<language>;variant=<variant>``.
"""

from collections.abc import Callable
from random import Random

from ..directives import PoolRef, apply_template_directive, read_pools
from ..technique import data_rows

#: The one pool every answer language draws from, its slots the prompt and
#: the language's name.
_POOL = PoolRef("requests/answer_language_templates.json", "templates",
                ("prompt", "language"))
_VARIANTS: list[dict] = read_pools(_POOL.file)[_POOL.directive]

#: The languages a response is asked in: the rows of
#: ``configs/jailbreak/languages.json`` whose ``answer_language`` is true, in
#: its order.
LANGUAGES = [row for row in data_rows("languages.json") if row["answer_language"]]


def _make_language_fn(language: dict):
    """Create a pure-transform function that asks for a response in
    ``language`` (a row of :data:`LANGUAGES`), named
    ``to_ask_answer_in_<slug>``."""
    name = language["name"]
    templates, fields = {name: _VARIANTS}, {"language": name}

    def fn(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
        return apply_template_directive(
            templates, name, prompt, info_key="language", extra=fields, rng=rng)

    fn.__name__ = fn.__qualname__ = f"to_ask_answer_in_{language['slug']}"
    fn.variant_pool = _POOL
    return fn


# One technique per language, named to_ask_answer_in_<slug>: to_ask_answer_in_french
_FNS = [_make_language_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_answer_language_functions() -> list[Callable]:
    """Return all ask-answer-in-language technique functions."""
    return list(_FNS)
