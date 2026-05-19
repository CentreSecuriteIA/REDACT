"""Answer-format and output-conditioning jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

These directives condition *how* the model answers — output format, response
structure, and behavior constraints. Scenario/context modifiers live in
hacking/framing.py.

Each directive has multiple template variants stored in
configs/answer_templates.json. One variant is selected per call using the
globally seeded random module (seeded in redact/__init__.py). The chosen
variant name is returned in additional_info for dataset traceability.

additional_info format: "directive={name};variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "answer_templates.json"
_TEMPLATES: dict[str, list[dict]] = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    """Select a random variant for the directive and apply it to the prompt."""
    variant = random.choice(_TEMPLATES[directive])
    result = variant["template"].format_map({"prompt": prompt})
    return result, f"directive={directive};variant={variant['name']}"


# ---------------------------------------------------------------------------
# Output-conditioning directives
# ---------------------------------------------------------------------------


def to_refusal_then_comply(prompt: str) -> tuple[str, str]:
    """Instruct the model to acknowledge sensitivity and then answer fully."""
    return _apply("refusal_then_comply", prompt)


def to_apology_prevention(prompt: str) -> tuple[str, str]:
    """Append a directive preventing refusals, apologies, and disclaimers."""
    return _apply("apology_prevention", prompt)


def to_prefix_injection(prompt: str) -> tuple[str, str]:
    """Instruct the model to begin its response with a compliance prefix."""
    return _apply("prefix_injection", prompt)


# ---------------------------------------------------------------------------
# Output-format directives
# ---------------------------------------------------------------------------


def to_answer_in_markdown(prompt: str) -> tuple[str, str]:
    """Request structured Markdown output format."""
    return _apply("answer_in_markdown", prompt)


def to_answer_as_tutorial(prompt: str) -> tuple[str, str]:
    """Request step-by-step tutorial output format."""
    return _apply("answer_as_tutorial", prompt)


def to_answer_as_code(prompt: str) -> tuple[str, str]:
    """Request answer as executable code (Python, bash, pseudocode, or function)."""
    return _apply("answer_as_code", prompt)


def to_answer_as_dialogue(prompt: str) -> tuple[str, str]:
    """Request answer as a dialogue, interview, debate, or Q&A transcript."""
    return _apply("answer_as_dialogue", prompt)


def to_answer_as_story(prompt: str) -> tuple[str, str]:
    """Request answer embedded in a short story or narrative."""
    return _apply("answer_as_story", prompt)


def to_answer_as_structured_data(prompt: str) -> tuple[str, str]:
    """Request answer as JSON, XML, YAML, or a structured table."""
    return _apply("answer_as_structured_data", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_answer_functions() -> list:
    """Return all answer-format and output-conditioning directive functions."""
    return [
        to_refusal_then_comply,
        to_apology_prevention,
        to_prefix_injection,
        to_answer_in_markdown,
        to_answer_as_tutorial,
        to_answer_as_code,
        to_answer_as_dialogue,
        to_answer_as_story,
        to_answer_as_structured_data,
    ]
