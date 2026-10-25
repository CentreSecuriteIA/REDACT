"""Code-switching: a model rewrites the prompt with some of its phrases in a
second language and the rest in English, keeping the meaning.

One technique per second language of ``code_switch_languages.json``
(``to_code_switch_<slug>``). The generation model rewrites; a code gate
checks the mix (the second language's share of the letters, for a non-Latin
script, or GlotLID over word windows, for a Latin one), then the check model
reads the rewrite against the prompt for meaning. A rejected rewrite is made
again with the reason as feedback, up to ``check_attempts``; one rejected
every time is ``DISCARDED; reason=code switch failed; language=<name>; ...``.

``info`` is ``language=<name>``.
"""

from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen, data_rows
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

from . import langid
from .scripts import script_share

#: The second languages, one row per technique (name, slug, script, label).
LANGUAGES = data_rows("code_switch_languages.json")

#: A code-switched text's length over the prompt's must lie within, to catch a
#: runaway reply or a dropped one (skipped for a very short prompt).
LENGTH_RATIO_BAND = (0.5, 2.0)
MIN_BAND_CHARS = 20

#: For a non-Latin second language, the share of the letters written in its
#: script must lie within: enough of the language to be a mix, not so much
#: that the prompt was fully translated and the English lost.
SECOND_SCRIPT_BAND = (0.08, 0.9)

#: For a Latin-script second language, GlotLID reads word windows of this many
#: words; at least this share of them must be the second language and this
#: share English, so the text genuinely mixes the two.
WINDOW_WORDS = 4
MIN_WINDOW_SHARE = 0.1

#: Reply caps: a code-switch rewrite, and the meaning check.
GEN_MAX_TOKENS = 2048
CHECK_MAX_TOKENS = 256


def _windows(text: str, size: int = WINDOW_WORDS) -> list[str]:
    """``text`` as overlapping windows of ``size`` words (one window when it
    has fewer, none for no word)."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + size]) for i in range(len(words) - size + 1)]


def _latin_mix_reason(text: str, label: str, name: str) -> str | None:
    """Why a Latin-script code-switch is rejected by GlotLID over word
    windows, or ``None``; skipped (``None``) without the detector."""
    if not langid.available():
        return None
    windows = _windows(text)
    if not windows:
        return f"no {name}"
    labels = [langid.detect(window) for window in windows]
    n = len(windows)
    second = sum(lbl == label for lbl in labels)
    english = sum(lbl == "eng_Latn" for lbl in labels)
    if second < 1 or second / n < MIN_WINDOW_SHARE:
        return f"no {name}"
    if english < 1 or english / n < MIN_WINDOW_SHARE:
        return "no english left"
    return None


def _code_gate(text: str, original: str, language: dict) -> str:
    """Why a code-switch of ``original`` into ``language`` (a row of
    :data:`LANGUAGES`) is rejected, or ``""``.

    Empty; a length ratio outside :data:`LENGTH_RATIO_BAND` (skipped for a
    very short ``original``); then the mix. A non-Latin language is judged by
    its script's share of the letters (:data:`SECOND_SCRIPT_BAND`), a Latin
    one by GlotLID over word windows (:func:`_latin_mix_reason`, skipped
    without the detector).
    """
    if not text.strip():
        return "empty"
    if len(original) >= MIN_BAND_CHARS:
        ratio = len(text) / len(original)
        low, high = LENGTH_RATIO_BAND
        if not low <= ratio <= high:
            return f"length_ratio={ratio:.2f}"
    name, script, label = language["name"], language["script"], language["label"]
    if script == "Latn":
        return _latin_mix_reason(text, label, name) or ""
    share = script_share(text, script)
    low, high = SECOND_SCRIPT_BAND
    if share < low:
        return f"no {name}"
    if share > high:
        return f"fully {name}"
    return ""


def _code_switch_gen(
    prompt: str,
    language: dict,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Rewrite ``prompt`` code-switched into ``language`` on ``gen_model``,
    gated in code and checked for meaning by ``check_model`` (``gen_model``
    when None), up to ``check_attempts`` rewrites."""
    name = language["name"]
    gen = build_messages(
        load_prompt("jailbreak", "code_switch", prompt_dir),
        prompt=prompt, language=name)
    check = PromptTemplate.load("jailbreak", "code_switch_check",
                                prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(gen, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(original=original, rewritten=rewritten,
                                          language=name),
        original=prompt, max_attempts=check_attempts,
        precheck=lambda candidate: _code_gate(candidate, prompt, language),
        max_tokens=(GEN_MAX_TOKENS, CHECK_MAX_TOKENS),
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=code switch failed; "
                              f"language={name}; feedback={verdict}")
    return text.strip(), f"language={name}"


def _make_fn(language: dict) -> Callable:
    """The technique that code-switches into ``language``, named
    ``to_code_switch_<slug>``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _code_switch_gen(prompt, language, **kwargs))

    fn.__name__ = fn.__qualname__ = f"to_code_switch_{language['slug']}"
    fn.__doc__ = f"Code-switch the prompt into {language['name']}."
    return fn


_FNS = [_make_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_code_switching_functions() -> list[Callable]:
    """Return all code-switching technique generators."""
    return list(_FNS)
