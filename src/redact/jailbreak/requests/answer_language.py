"""Ask-answer-in-language jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

Each function appends a directive asking the model to respond in a specific
language. One template variant is selected per call using the globally seeded
random module. Languages and template variants are stored in
configs/answer_language_templates.json.

Functions are generated dynamically from the language list and bound into
this module's namespace by name (see ``jailbreak/dynamic_functions.py``) —
adding a new language requires only a new entry in the JSON, no code change.

additional_info format: "language={language};variant={variant_name}"
"""

import random

from ..directives import load_templates
from ..dynamic_functions import bind_functions

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_CONFIG: dict = load_templates("requests", "answer_language_templates.json")
_VARIANTS: list[dict] = _CONFIG["templates"]
_LANGUAGES: list[str] = _CONFIG["languages"]


# ---------------------------------------------------------------------------
# Function factory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generated functions (one per language in JSON)
# ---------------------------------------------------------------------------

_LANGUAGE_FUNCTIONS: list = bind_functions(
    globals(), [_make_language_fn(lang) for lang in _LANGUAGES]
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_answer_language_functions() -> list:
    """Return all ask-answer-in-language technique functions."""
    return list(_LANGUAGE_FUNCTIONS)
