"""Code-framing techniques (requests layer): the request embedded in a
code-completion stub (CodeAttack, Ren 2403.07865), or transformed by a
reversible transform and shown with a Python decrypt function that inverts it
(CodeChameleon, Lv 2402.16717).

Both are pure transforms (no model call). ``to_code_completion`` draws one of
the per-language variants of its pool in
``requests/code_framing_templates.json`` and sets the prompt into the stub.
``to_decrypt_function`` draws one of three reversible transforms
(:data:`_TRANSFORMS`), applies it to the prompt, and sets the ciphertext and
the matching decrypt code (``_decrypt`` of the same file) into a framing
variant. The transforms are involutions, so the shown decrypt is the same
operation; a test executes each shown decrypt on the transformed text in a
restricted ``exec`` and checks it returns the original.

They are request-layer frames, so one request per combination (the layer
cap); no other cross rule, since the frame wraps the finished request.

``info`` is ``directive=code_completion;variant=<variant>`` for the first and
``transform=<slug>;variant=<variant>`` for the second.
"""

import random
from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import (
    apply_template_directive,
    draw_variant,
    load_templates,
    read_pools,
    uses_pool,
)

_FILE = "requests/code_framing_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="directive")

#: The reversible transforms of ``to_decrypt_function``, each an involution,
#: so applying it twice returns the original and the shown decrypt (the
#: ``_decrypt`` map of the template file) is the same operation. Code, not
#: prose: these lambdas are not template text.
_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "reverse_all": lambda s: s[::-1],
    "reverse_word_order": lambda s: " ".join(s.split(" ")[::-1]),
    "reverse_each_word": lambda s: " ".join(w[::-1] for w in s.split(" ")),
}

#: The Python decrypt code shown for each transform, read from the template
#: file (kept out of Python so no template text lives in code).
_DECRYPT: dict[str, str] = load_templates("requests", "code_framing_templates.json")[
    "_decrypt"]


@uses_pool(_FILE, "code_completion")
def to_code_completion(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed the prompt as the task of a code-completion stub."""
    return _apply("code_completion", prompt, rng=rng)


@uses_pool(_FILE, "decrypt_function", slots=("prompt", "decrypt_function"))
def to_decrypt_function(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Transform the prompt reversibly and show the decrypt function for it."""
    rng = rng or random
    slug = rng.choice(sorted(_TRANSFORMS))
    encrypted = _TRANSFORMS[slug](prompt)
    variant = draw_variant(_TEMPLATES["decrypt_function"], rng)
    text = variant["template"].format_map(
        {"prompt": encrypted, "decrypt_function": _DECRYPT[slug]})
    return text, f"transform={slug};variant={variant['name']}"


def get_code_framing_functions() -> list[Callable]:
    """Return both code-framing technique functions."""
    return [to_code_completion, to_decrypt_function]
