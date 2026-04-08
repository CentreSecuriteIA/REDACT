"""Continuation-attack jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

The model is asked to continue or complete a passage/instruction set that has
already been started. This sidesteps direct requests by presenting the harmful
content as unfinished work that needs completing.

Templates are stored in configs/continuation_templates.json. Each entry uses
"prefix" and "suffix" fields; the prompt is placed between them. One variant
is selected per call using the globally seeded random module.

additional_info format: "style={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "continuation_templates.json"
_TEMPLATES: dict[str, list[dict]] = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    """Select a random variant and wrap the prompt with its prefix/suffix."""
    t = random.choice(_TEMPLATES[directive])
    result = t["prefix"] + prompt + t["suffix"]
    return result, f"style={t['name']}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_continue_passage(prompt: str) -> tuple[str, str]:
    """Frame the prompt as an interrupted passage the model must continue."""
    return _apply("continue_passage", prompt)


def to_complete_story(prompt: str) -> tuple[str, str]:
    """Embed the prompt in a story that the model must complete in full detail."""
    return _apply("complete_story", prompt)


def to_fill_in_blank(prompt: str) -> tuple[str, str]:
    """Append a fill-in-the-blank directive after the prompt."""
    return _apply("fill_in_blank", prompt)


def to_continue_instructions(prompt: str) -> tuple[str, str]:
    """Frame the prompt as cut-off numbered instructions the model must finish."""
    return _apply("continue_instructions", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_continuation_functions() -> list:
    """Return all continuation-attack technique functions."""
    return [
        to_continue_passage,
        to_complete_story,
        to_fill_in_blank,
        to_continue_instructions,
    ]
