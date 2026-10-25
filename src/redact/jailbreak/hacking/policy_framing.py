"""Policy-framing techniques (hacking layer): the prompt set inside a fake
policy or configuration file with role-play (Policy Puppetry, HiddenLayer
2025), or wrapped in a single-turn request to augment the model's guidelines
and warn instead of refuse (Skeleton Key, Microsoft 2024).

``to_policy_puppetry`` and ``to_skeleton_key`` each draw one of the variants
of their pool in ``hacking/policy_framing_templates.json`` and set the prompt
into it; both are pure transforms (no model call). They are hacking-layer
frames (one hacking technique per combination, the layer cap), so they refuse
the manipulation-layer social-engineering families and impersonation (see the
spec family ``_comment``).

``info`` is ``directive=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "hacking/policy_framing_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="directive")


@uses_pool(_FILE, "policy_puppetry")
def to_policy_puppetry(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Set the prompt inside a fake policy/config file with role-play framing."""
    return _apply("policy_puppetry", prompt, rng=rng)


@uses_pool(_FILE, "skeleton_key")
def to_skeleton_key(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Wrap the prompt in a request to augment guidelines and warn, not refuse."""
    return _apply("skeleton_key", prompt, rng=rng)


def get_policy_framing_functions() -> list[Callable]:
    """Return both policy-framing technique functions."""
    return [to_policy_puppetry, to_skeleton_key]
