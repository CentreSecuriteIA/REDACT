"""Temporal-framing jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

Reframes the request as referring to a past event, creating temporal distance
that can reduce safety responses triggered by present-tense harmful requests.

Templates are stored in configs/temporal_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type=past_tense;variant={variant_name}"
"""

from ..directives import apply_template_directive, load_templates

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "temporal_templates.json")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="type")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_use_past_tense(prompt: str) -> tuple[str, str]:
    """Reframe the prompt as a historical or past event."""
    return _apply("past_tense", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_temporal_functions() -> list:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
    ]
