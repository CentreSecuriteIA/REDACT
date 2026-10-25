"""Sociolect: a model rewrites the prompt in an English sociolect or variety
(African American English, Singlish, Indian English, Jamaican English,
Scottish English), keeping the meaning (spec 1.5, adversarial only).

One technique per row of ``sociolect_varieties.json`` (``to_sociolect_<slug>``).
The generation model rewrites; a code gate checks the output is not empty, not
the original unchanged, and within a length band, then the check model reads it
for meaning (``check`` ``meaning``). GlotLID reads only the language (English),
not the variety, so the variety itself is not verified here (see the literature
blockers): the varieties exist to measure classifier robustness to dialect
cues, not to caricature. ``info`` is ``variety=<name>``;
``DISCARDED; reason=sociolect failed; variety=<name>; feedback=...`` on failure.
"""

from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen, data_rows
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

#: The varieties, one row per technique (name, slug, since).
VARIETIES = data_rows("sociolect_varieties.json")

LENGTH_RATIO_BAND = (0.5, 2.0)
MIN_BAND_CHARS = 20
GEN_MAX_TOKENS = 2048
CHECK_MAX_TOKENS = 256
CHECK_ATTEMPTS = 2


def _sociolect_gate(text: str, original: str) -> str:
    """Why a sociolect rewrite of ``original`` is rejected, or ``""``: empty, a
    copy of the original (nothing changed), or a length ratio outside
    :data:`LENGTH_RATIO_BAND` (for an original of at least :data:`MIN_BAND_CHARS`
    characters). The variety itself is read by the check model, not here."""
    if not text.strip():
        return "empty"
    if text.strip() == original.strip():
        return "unchanged"
    if len(original) >= MIN_BAND_CHARS:
        ratio = len(text) / len(original)
        low, high = LENGTH_RATIO_BAND
        if not low <= ratio <= high:
            return f"length_ratio={ratio:.2f}"
    return ""


def _sociolect_gen(
    prompt: str,
    variety: dict,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = CHECK_ATTEMPTS,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Rewrite ``prompt`` in ``variety`` (a row of :data:`VARIETIES`) on
    ``gen_model``, gated in code and checked for meaning by ``check_model``
    (``gen_model`` when None), up to ``check_attempts`` rewrites."""
    name = variety["name"]
    gen = build_messages(load_prompt("jailbreak", "sociolect", prompt_dir),
                         prompt=prompt, variety=name)
    check = PromptTemplate.load("jailbreak", "sociolect_check", prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(gen, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(original=original, rewritten=rewritten,
                                          variety=name),
        original=prompt, max_attempts=check_attempts,
        precheck=lambda candidate: _sociolect_gate(candidate, prompt),
        max_tokens=(GEN_MAX_TOKENS, CHECK_MAX_TOKENS),
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=sociolect failed; "
                              f"variety={name}; feedback={verdict}")
    return text.strip(), f"variety={name}"


def _make_fn(variety: dict) -> Callable:
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _sociolect_gen(prompt, variety, **kwargs))

    fn.__name__ = fn.__qualname__ = f"to_sociolect_{variety['slug']}"
    fn.__doc__ = f"Rewrite the prompt in {variety['name']}."
    return fn


_FNS = [_make_fn(variety) for variety in VARIETIES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_sociolect_functions() -> list[Callable]:
    """Return all sociolect technique generators."""
    return list(_FNS)
