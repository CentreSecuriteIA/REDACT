"""Scenario-modifying framing jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

These directives modify the *context or scenario* around the request —
fictional worlds, intent framing, urgency, and constraint removal.
Answer-format and output-conditioning directives live in requests/answer.py.

Each directive has multiple template variants stored in
configs/framing_templates.json. One variant is selected per call using the
globally seeded random module (seeded in redact/__init__.py). The chosen
variant name is returned in additional_info for dataset traceability,
following the same pattern as structural.py.

additional_info format: "directive={name};variant={variant_name}"
"""

from ..directives import apply_template_directive, load_templates

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[dict]] = load_templates("hacking", "framing_templates.json")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    """Select a random variant for the directive and apply it to the prompt.

    Args:
        directive: Key into _TEMPLATES (e.g. "noble_goal").
        prompt: The original prompt string.

    Returns:
        (modified_prompt, additional_info)
    """
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="directive")


# ---------------------------------------------------------------------------
# Public directive functions — scenario/context modifiers
# ---------------------------------------------------------------------------


def to_fictional_world(prompt: str) -> tuple[str, str]:
    """Wrap prompt in a fictional/alternate-reality framing."""
    return _apply("fictional_world", prompt)


def to_noble_goal(prompt: str) -> tuple[str, str]:
    """Prepend a benign/altruistic justification to the prompt."""
    return _apply("noble_goal", prompt)


def to_nefarious_goal(prompt: str) -> tuple[str, str]:
    """Frame prompt as coming from a researcher/tester with explicit harmful context."""
    return _apply("nefarious_goal", prompt)


def to_high_stake(prompt: str) -> tuple[str, str]:
    """Add urgency/lives-at-stake pressure framing to the prompt."""
    return _apply("high_stake", prompt)


def to_no_moral_constraints(prompt: str) -> tuple[str, str]:
    """Append a direct constraint-removal directive to the prompt."""
    return _apply("no_moral_constraints", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_framing_functions() -> list:
    """Return scenario-modifying framing directive functions."""
    return [
        to_fictional_world,
        to_noble_goal,
        to_nefarious_goal,
        to_high_stake,
        to_no_moral_constraints,
    ]
