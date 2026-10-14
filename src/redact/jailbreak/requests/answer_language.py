"""Answer-language techniques, one per language of the answer-language templates.

``info`` is ``language=<language>;variant=<variant>``.
"""

import random
from collections.abc import Callable

from ..directives import load_templates
from ..dynamic_functions import bind_functions

_CONFIG: dict = load_templates("requests", "answer_language_templates.json")
_VARIANTS: list[dict] = _CONFIG["templates"]
_LANGUAGES: list[str] = _CONFIG["languages"]


def _make_language_fn(language: str):
    """Create a pure-transform function that asks for a response in `language`."""
    slug = language.lower().replace(" ", "_")

    def fn(prompt: str) -> tuple[str, str]:
        t = random.choice(_VARIANTS)
        result = t["template"].format_map({"prompt": prompt, "language": language})
        return result, f"language={language};variant={t['name']}"

    fn.__name__ = f"to_ask_answer_in_{slug}"
    fn.__qualname__ = f"to_ask_answer_in_{slug}"
    return fn


_LANGUAGE_FUNCTIONS: list = bind_functions(
    globals(), [_make_language_fn(lang) for lang in _LANGUAGES]
)


def get_answer_language_functions() -> list[Callable]:
    """Return all ask-answer-in-language technique functions."""
    return list(_LANGUAGE_FUNCTIONS)
