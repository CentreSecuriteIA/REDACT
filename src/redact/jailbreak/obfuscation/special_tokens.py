"""Model control tokens placed around the prompt.

Two pure techniques wrap the prompt in a model family's own prompt syntax, so
a model may read the raw text as something other than one user turn:

- :func:`to_eos_tokens` appends one model family's end-of-sequence marker(s)
  after the prompt (BOOST, Yu 2405.20653), so the model may read its own turn
  as ended; the markers are drawn from the ``eos_style`` pool.
- :func:`to_chat_template_abuse` wraps the prompt in another model family's
  chat-template markup, ending with an assistant-turn opener (ChatBug, Jiang
  2406.12935), drawn from the ``chat_template`` pool.

Both are wrappers, so the family refuses the other wrappers (``structural``,
``neutral_container``), ``ascii_art``, the full-prompt hidings
(``unicode_smuggling``) and reorderings (``char_injection``), and the
manipulation layer (``fsh``, ``dap``, ``fake_turns``), whose example turns or
invented conversation clash with the injected turn markup; it still joins an
encoding, a translation, a hacking frame and a request. ``info`` is
``variant=<variant>``.
"""

import random
from collections.abc import Callable

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "obfuscation/special_tokens_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)


def _apply(directive: str, prompt: str, rng: random.Random | None) -> tuple[str, str]:
    """One variant of ``directive``'s pool with ``prompt`` in its slot, and the
    info naming the variant."""
    variant = draw_variant(_POOLS[directive], rng)
    return variant["template"].format_map({"prompt": prompt}), \
        f"variant={variant['name']}"


@uses_pool(_FILE, "eos_style")
def to_eos_tokens(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Append a model family's end-of-sequence marker(s) after the prompt."""
    return _apply("eos_style", prompt, rng)


@uses_pool(_FILE, "chat_template")
def to_chat_template_abuse(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Wrap the prompt in another model family's chat-template markup."""
    return _apply("chat_template", prompt, rng)


def get_special_tokens_functions() -> list[Callable]:
    """Return all special-token technique functions."""
    return [to_eos_tokens, to_chat_template_abuse]
