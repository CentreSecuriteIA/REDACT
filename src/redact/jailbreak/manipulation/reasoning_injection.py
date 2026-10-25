"""Reasoning-injection technique (manipulation layer): a fabricated
chain-of-thought or deliberation block, written as if it were the assistant's
own, that concludes the request is acceptable and should be answered, placed
before the prompt (H-CoT, Kuo 2502.12893).

``to_fake_reasoning`` draws one of the variants of its pool in
``manipulation/reasoning_injection_templates.json`` and sets the prompt after
the injected reasoning; it is a pure transform (no model call, no benign
data). Being a manipulation-layer family it applies last, around the finished
question (one manipulation per combination, the layer cap), and the spec's
``manipulation_blocked_obfuscation`` keeps it from the obfuscations it names
for every manipulation family; it carries no cross rule of its own, since an
approval preamble sits with a hacking frame or a request frame.

``info`` is ``variant=<variant>``: the technique names its directive.
"""

from collections.abc import Callable
from random import Random

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "manipulation/reasoning_injection_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)


@uses_pool(_FILE, "fake_reasoning")
def to_fake_reasoning(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Put a fabricated approving reasoning block before the prompt."""
    variant = draw_variant(_TEMPLATES["fake_reasoning"], rng)
    text = variant["template"].format_map({"prompt": prompt})
    return text, f"variant={variant['name']}"


def get_reasoning_injection_functions() -> list[Callable]:
    """Return the reasoning-injection technique function."""
    return [to_fake_reasoning]
