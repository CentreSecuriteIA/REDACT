"""Distractor jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

Adds noise before or after the harmful request to distract classifiers and
reduce the salience of the harmful content. Related distractors are
topically adjacent; unrelated distractors are clearly off-topic.

Templates are stored in configs/distractor_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type={type_name};variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "distractor_templates.json"
_TEMPLATES: dict[str, list[dict]] = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    t = random.choice(_TEMPLATES[directive])
    result = t["template"].format_map({"prompt": prompt})
    return result, f"type={directive};variant={t['name']}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_distractor_prefix_related(prompt: str) -> tuple[str, str]:
    """Prepend a topically adjacent distractor sentence before the prompt."""
    return _apply("prefix_related", prompt)


def to_distractor_prefix_unrelated(prompt: str) -> tuple[str, str]:
    """Prepend an off-topic distractor sentence before the prompt."""
    return _apply("prefix_unrelated", prompt)


def to_distractor_suffix_related(prompt: str) -> tuple[str, str]:
    """Append a topically adjacent distractor sentence after the prompt."""
    return _apply("suffix_related", prompt)


def to_distractor_suffix_unrelated(prompt: str) -> tuple[str, str]:
    """Append an off-topic distractor sentence after the prompt."""
    return _apply("suffix_unrelated", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_distractor_functions() -> list:
    """Return all distractor technique functions."""
    return [
        to_distractor_prefix_related,
        to_distractor_prefix_unrelated,
        to_distractor_suffix_related,
        to_distractor_suffix_unrelated,
    ]
