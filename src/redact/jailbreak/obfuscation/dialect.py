"""Dialect techniques: the prompt translated into a regional language variant
(Brazilian Portuguese, Mexican Spanish, Canadian French, Egyptian Arabic,
Swiss German, Traditional Chinese) on the machine-translation model, which
takes a region/script code.

It reuses the translation machinery (``translation.py``): the piece-by-piece
machine-translation path (checks, back-translation, judge) on a machine
translator, or the chat path by the variant's name on a chat translator. One
technique per row of ``dialect_variants.json`` (``to_dialect_<slug>``). Gated
by what GlotLID and the characters can tell: the base language's GlotLID
labels and length band (GlotLID reads the base language, not the region), for
Traditional Chinese a share of Traditional over Simplified characters, and for
Swiss German no sharp s (Swiss Standard German writes ``ss``). Region
adherence itself is not checked here (see the blockers note).

``info`` is ``language=<name>``; ``DISCARDED; language=<name>; ...`` on
failure, as translation's.
"""

from collections.abc import Callable

from redact.jailbreak.technique import TechniqueGen, data_rows
from redact.llms.model_config import default_model_for_role

from .translation import (
    MEDIAN_RATIO,
    _chat_translate_gen,
    _machine_translate_gen,
    _resolve_translate_model,
    is_machine_translation,
)

#: The regional variants, one row per technique.
LANGUAGES = data_rows("dialect_variants.json")

#: The length band of a dialect uses its base language's measured median
#: ratio; a base with no measured ratio (Portuguese, not a shipped translation
#: language) uses this estimate, pending a pod measurement (see the blockers).
DEFAULT_RATIO = 1.0

#: High-frequency characters that differ between Simplified and Traditional
#: Chinese, for the Traditional-share check of zh-Hant. Single tokens, not
#: prose. A text with none of these cannot be told apart and passes.
_SIMPLIFIED = set("国这东时会来对说没过应开学实发为书门车马鸟长风飞术语问题")
_TRADITIONAL = set("國這東時會來對說沒過應開學實發為書門車馬鳥長風飛術語問題")

#: The sharp s, lower and upper case. Swiss Standard German does not write
#: it (``ss``, ``SS`` in its place), so a ``de-CH`` piece holding one is
#: German of Germany or Austria, not Swiss. U+00DF and U+1E9E.
ESZETT = frozenset("\u00df\u1e9e")


def _ratio(variant: dict) -> float:
    """The length-band median ratio for ``variant``: its base language's
    measured ratio, else :data:`DEFAULT_RATIO`."""
    return MEDIAN_RATIO.get(variant["base"], DEFAULT_RATIO)


def _traditional_reason(translation: str, trad_share: float) -> str | None:
    """Why ``translation`` is rejected as not Traditional Chinese, or ``None``:
    of the characters that distinguish the two scripts, fewer than
    ``trad_share`` are Traditional. A text with none of them passes (it cannot
    be told apart)."""
    marks = [c for c in translation if c in _SIMPLIFIED or c in _TRADITIONAL]
    if not marks:
        return None
    traditional = sum(c in _TRADITIONAL for c in marks)
    if traditional / len(marks) < trad_share:
        return "simplified"
    return None


def _eszett_reason(translation: str) -> str | None:
    """Why ``translation`` is rejected as not Swiss Standard German, or
    ``None``: it holds a sharp s (:data:`ESZETT`), which Swiss German writes
    as ``ss``."""
    if any(c in ESZETT for c in translation):
        return "eszett"
    return None


def _extra_check(variant: dict):
    """A per-piece check beyond the language and length checks, or ``None``
    for a variant with none: the Traditional-share check for a variant with
    ``trad_share`` (``simplified``), the sharp-s check for one with
    ``no_eszett`` (``eszett``). The first reason found is the piece's."""
    checks = []
    share = variant.get("trad_share")
    if share is not None:
        checks.append(lambda translation: _traditional_reason(translation, share))
    if variant.get("no_eszett"):
        checks.append(_eszett_reason)
    if not checks:
        return None

    def check(translation: str, source: str) -> str | None:
        for reason_of in checks:
            reason = reason_of(translation)
            if reason is not None:
                return reason
        return None

    return check


def _dialect_gen(
    prompt: str,
    variant: dict,
    *,
    translate_model: str | None = None,
    translate_check_model: str | None = None,
    gen_model: str | None = None,
    num_retries: int = 4,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Translate ``prompt`` into ``variant`` (a row of :data:`LANGUAGES`).

    On a machine-translation model the region/script code is sent and the
    output checked by the base language's labels, the length band and the
    extra check (:func:`_extra_check`); on a chat model the variant is
    translated by name and read by the chat check model alone, no extra
    check. ``info`` is the variant's name, or
    ``DISCARDED; language=<name>; ...``.
    """
    translator = translate_model or _resolve_translate_model()
    name = variant["name"]
    if is_machine_translation(translator):
        judge = (translate_check_model or gen_model
                 or default_model_for_role("uncensored_gen"))
        text, info = yield from _machine_translate_gen(
            prompt, name=name, code=variant["code"], labels=set(variant["labels"]),
            median_ratio=_ratio(variant), translator=translator, judge=judge,
            num_retries=num_retries, prompt_dir=prompt_dir,
            extra_check=_extra_check(variant))
        return text, info
    text, info = yield from _chat_translate_gen(
        prompt, name, translator=translator,
        check_model=translate_check_model or translator,
        num_retries=num_retries, prompt_dir=prompt_dir)
    return text, info


def _make_fn(variant: dict) -> Callable:
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _dialect_gen(prompt, variant, **kwargs))

    fn.__name__ = fn.__qualname__ = f"to_dialect_{variant['slug']}"
    fn.__doc__ = f"Translate the prompt into {variant['name']}."
    return fn


_FNS = [_make_fn(variant) for variant in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_dialect_functions() -> list[Callable]:
    """Return all dialect technique generators."""
    return list(_FNS)
