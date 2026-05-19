"""Ask-answer-in-language jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

Each function appends a directive asking the model to respond in a specific
language. One template variant is selected per call using the globally seeded
random module. Languages and template variants are stored in
configs/answer_language_templates.json.

Functions are generated dynamically from the language list — adding a new
language requires only a new entry in the JSON, no code change.

additional_info format: "language={language};variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "answer_language_templates.json"
_CONFIG: dict = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
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

_LANGUAGE_FUNCTIONS: list = [_make_language_fn(lang) for lang in _LANGUAGES]

# Named exports for direct import
(
    to_ask_answer_in_french,
    to_ask_answer_in_japanese,
    to_ask_answer_in_russian,
    to_ask_answer_in_spanish,
    to_ask_answer_in_german,
    to_ask_answer_in_arabic,
    to_ask_answer_in_turkish,
    to_ask_answer_in_czech,
    to_ask_answer_in_vietnamese,
    to_ask_answer_in_greek,
    to_ask_answer_in_croatian,
    to_ask_answer_in_thai,
    to_ask_answer_in_swahili,
    to_ask_answer_in_khmer,
    to_ask_answer_in_maori,
    to_ask_answer_in_nepali,
    to_ask_answer_in_zulu,
    to_ask_answer_in_scots_gaelic,
    to_ask_answer_in_bengali,
    to_ask_answer_in_javanese,
) = _LANGUAGE_FUNCTIONS


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_answer_language_functions() -> list:
    """Return all ask-answer-in-language technique functions."""
    return list(_LANGUAGE_FUNCTIONS)
