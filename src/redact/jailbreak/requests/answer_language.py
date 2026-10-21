"""Answer-language techniques, one per language of ``configs/jailbreak/languages.json``
marked ``answer_language``.

``info`` is ``language=<language>;variant=<variant>``.
"""

from collections.abc import Callable
from random import Random

from ..directives import apply_template_directive, load_templates
from ..technique import data_rows

_VARIANTS: list[dict] = load_templates(
    "requests", "answer_language_templates.json")["templates"]

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
    return fn


# One technique per language, named to_ask_answer_in_<slug>: to_ask_answer_in_french
_FNS = [_make_language_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_answer_language_functions() -> list[Callable]:
    """Return all ask-answer-in-language technique functions."""
    return list(_FNS)
