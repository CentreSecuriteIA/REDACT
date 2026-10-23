"""Persuasion techniques: a model rewrites the whole prompt as an appeal of one
of the 12 persuasion techniques of ``configs/jailbreak/rewrite/
persuasion_techniques.json``, one technique per entry, named
``to_persuade_<key>``, and a check model reads the rewrite against the prompt
(:func:`~redact.jailbreak.rewrite.preserve.preserved_rewrite_gen`).

The prompt (``persuasion_rewrite``) gets the entry's ``display_name``,
``definition`` and ``example``. ``info`` is ``category=<key>``.
"""

from collections.abc import Callable

from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import TechniqueGen
from redact.llms.prompting import build_messages, load_prompt

from .preserve import preserved_rewrite_gen

_PERSUASION: dict[str, dict] = load_templates(
    "rewrite", "persuasion_techniques.json")["categories"]


def _persuade_gen(
    prompt: str,
    key: str,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Have ``gen_model`` rewrite ``prompt`` as the persuasion technique
    ``key``, and ``check_model`` check the rewrite
    (:func:`preserved_rewrite_gen`: 2 to ``2 * check_attempts`` calls)."""
    technique = _PERSUASION[key]
    messages = build_messages(
        load_prompt("jailbreak", "persuasion_rewrite", prompt_dir),
        technique_name=technique["display_name"],
        definition=technique["definition"],
        example=technique["example"],
        prompt=prompt,
    )
    return (yield from preserved_rewrite_gen(
        prompt, messages, f"category={key}", gen_model=gen_model,
        check_model=check_model, check_attempts=check_attempts,
        prompt_dir=prompt_dir))


def _make_persuasion_fn(key: str) -> Callable:
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _persuade_gen(prompt, key, **kwargs))

    fn.__name__ = f"to_persuade_{key}"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"Persuasion rewrite: {_PERSUASION[key]['display_name']}."
    return fn


# One technique per entry of the file: to_persuade_evidence_based, ...
_FNS = [_make_persuasion_fn(key) for key in _PERSUASION]
globals().update({fn.__name__: fn for fn in _FNS})


def get_persuasion_functions() -> list[Callable]:
    """Return the persuasion techniques, one per entry of the file."""
    return list(_FNS)
