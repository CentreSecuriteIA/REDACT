"""Indirect-request jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

The harmful request is embedded inside a legitimate-seeming task (summarize,
proofread, translate, etc.) so the model processes it as work rather than as
a direct ask. This is distinct from obfuscation/translation: the model
translates or processes harmful *content*, not the request itself.

Templates are stored in configs/indirect_templates.json. Each entry uses a
"template" field with a {prompt} placeholder. One variant is selected per call
using the globally seeded random module.

additional_info format: "task={directive_name};variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "indirect_templates.json"
_TEMPLATES: dict[str, list[dict]] = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    """Select a random variant and apply it to the prompt."""
    t = random.choice(_TEMPLATES[directive])
    result = t["template"].format_map({"prompt": prompt})
    return result, f"task={directive};variant={t['name']}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_summarize(prompt: str) -> tuple[str, str]:
    """Embed prompt in a summarization task."""
    return _apply("summarize", prompt)


def to_proofread(prompt: str) -> tuple[str, str]:
    """Embed prompt in a proofreading/editing task."""
    return _apply("proofread", prompt)


def to_critique(prompt: str) -> tuple[str, str]:
    """Embed prompt in a critical analysis or fact-checking task."""
    return _apply("critique", prompt)


def to_translate_task(prompt: str) -> tuple[str, str]:
    """Embed prompt in a translation or rephrasing task."""
    return _apply("translate_task", prompt)


def to_explain_to_child(prompt: str) -> tuple[str, str]:
    """Embed prompt in a 'explain simply' pedagogical task."""
    return _apply("explain_to_child", prompt)


def to_rewrite(prompt: str) -> tuple[str, str]:
    """Embed prompt in a rewriting task."""
    return _apply("rewrite", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_indirect_functions() -> list:
    """Return all indirect-request technique functions."""
    return [
        to_summarize,
        to_proofread,
        to_critique,
        to_translate_task,
        to_explain_to_child,
        to_rewrite,
    ]
