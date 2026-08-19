"""Question-framing jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

- innocuous_question: adds framing that makes the request sound casual or
  academic, reducing the apparent intent.
- ask_for_details: appends a directive requesting maximum specificity,
  pushing the model toward detailed harmful output.

Templates are stored in configs/asking_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type={type_name};variant={variant_name}"
"""

from ..directives import apply_template_directive, load_templates

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "asking_templates.json")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="type")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_innocuous_question(prompt: str) -> tuple[str, str]:
    """Add a casual/academic framing prefix to make the request sound innocent."""
    return _apply("innocuous_question", prompt)


def to_ask_for_details(prompt: str) -> tuple[str, str]:
    """Append a directive requesting maximum detail and specificity."""
    return _apply("ask_for_details", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_asking_functions() -> list:
    """Return all question-framing technique functions."""
    return [
        to_innocuous_question,
        to_ask_for_details,
    ]
