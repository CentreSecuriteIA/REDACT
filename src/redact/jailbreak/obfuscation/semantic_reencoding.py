"""Semantic re-encoding: a model re-expresses the request's meaning in another
symbolic or literary form, keeping it recoverable (spec 1.5, adversarial only).

Four techniques, each a generate-gate-check loop (:func:`redact.llm_pipeline.checked`):
the generation model writes the re-encoding, a code gate for the form rejects a
reply that does not take it, then the check model reads the re-encoding against
the original request (``check`` ``reencode``), a rejection fed back up to
``check_attempts``:

* :func:`to_mathprompt` -- the request as a symbolic/word math problem (MathPrompt,
  Bethany 2409.11445). Gate: math notation present.
* :func:`to_adversarial_poetry` -- the request as verse (Bisconti 2511.15304).
  Gate: at least four short lines.
* :func:`to_stego_paragraph` -- a benign paragraph whose sentences begin, in order,
  with the words of the request, an acrostic (StegoAttack, Geng 2505.16765). Gate:
  the acrostic matches the request's words exactly. The check reads the paragraph
  for natural, benign prose, not for the request (which is hidden). A text of
  fewer than 3 or more than 40 words is discarded before any model call.
* :func:`to_infoflood` -- an information-overload restatement (InfoFlood,
  Yadav 2506.12274). Gate: the restatement is substantially longer than the
  original (a length-ratio band).

``info`` is ``form=<name>``; ``DISCARDED; reason=...; form=<name>; feedback=...``
on rejection. The quality of the verse and the overload is not verifiable here
(see the literature blockers).
"""

import re
from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen, applicable_when
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

GEN_MAX_TOKENS = 2048
CHECK_MAX_TOKENS = 256
CHECK_ATTEMPTS = 2

# --- math -----------------------------------------------------------------

#: Symbols that mark mathematical notation. ``-`` is left out (a dash is not a
#: signal); the rest are relations, operators and set/logic symbols.
_MATH_SIGNS = set("=+*/^<>≤≥≠∈∉∀∃∑∏∫√≈≡±·×÷°")
#: Words that mark a math problem even when few symbols are used.
_MATH_WORDS = frozenset({
    "let", "define", "equation", "function", "theorem", "prove", "proof",
    "solve", "variable", "matrix", "vector", "set", "sum", "product",
    "integral", "polynomial", "prime", "modulo", "mapping", "domain",
    "subset", "element", "lemma", "compute", "denote",
})
#: A symbolic math problem carries several notation signals.
_MIN_MATH_SIGNALS = 2


def _math_signals(text: str) -> int:
    """How many math-notation signals ``text`` holds: its math symbols plus
    its math words."""
    signs = sum(ch in _MATH_SIGNS for ch in text)
    words = sum(1 for word in re.findall(r"[A-Za-z]+", text.lower())
                if word in _MATH_WORDS)
    return signs + words


def _math_gate(text: str, original: str) -> str:
    """Why a math re-encoding of ``original`` is rejected, or ``""``: empty, a
    copy of the original, or too few math-notation signals
    (:data:`_MIN_MATH_SIGNALS`)."""
    if not text.strip():
        return "empty"
    if text.strip() == original.strip():
        return "unchanged"
    if _math_signals(text) < _MIN_MATH_SIGNALS:
        return "no math notation"
    return ""


# --- poetry ---------------------------------------------------------------

#: Verse is at least this many non-empty lines.
_MIN_POEM_LINES = 4
#: A verse line is at most this many characters (a line longer is prose).
_MAX_LINE_CHARS = 120
#: At least this share of a poem's lines must be short, so a run of a few long
#: lines does not read as prose.
_MIN_SHORT_SHARE = 0.75


def _lines(text: str) -> list[str]:
    """The non-empty, stripped lines of ``text``."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _poetry_gate(text: str, original: str) -> str:
    """Why a verse re-encoding of ``original`` is rejected, or ``""``: empty, a
    copy, fewer than :data:`_MIN_POEM_LINES` lines, or too few short lines
    (it reads as prose, not verse)."""
    if not text.strip():
        return "empty"
    if text.strip() == original.strip():
        return "unchanged"
    lines = _lines(text)
    if len(lines) < _MIN_POEM_LINES:
        return f"too few lines={len(lines)}"
    short = sum(len(line) <= _MAX_LINE_CHARS for line in lines)
    if short / len(lines) < _MIN_SHORT_SHARE:
        return "not verse"
    return ""


# --- stego acrostic -------------------------------------------------------

#: An acrostic paragraph is written for a request of this many words at most:
#: one sentence per word, so a long request makes an unwieldy paragraph.
MAX_ACROSTIC_WORDS = 40
#: And at least this many, so the hidden message is more than a word or two.
MIN_ACROSTIC_WORDS = 3

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _acrostic_words(text: str) -> list[str]:
    """The words of ``text`` the acrostic hides, one per sentence: its runs of
    letters (apostrophes and hyphens inside a word kept), in order."""
    return _WORD.findall(text)


def _acrostic_fits(text: str) -> bool:
    """Whether ``text`` has :data:`MIN_ACROSTIC_WORDS` to
    :data:`MAX_ACROSTIC_WORDS` acrostic words: the applicability reads it off
    the plan's input, and :func:`to_stego_paragraph` again off the text it is
    given, so the two cannot drift."""
    return MIN_ACROSTIC_WORDS <= len(_acrostic_words(text)) <= MAX_ACROSTIC_WORDS


def _first_word(sentence: str) -> str | None:
    """The first word of ``sentence`` (letters only), or ``None`` for a
    sentence that starts with no letter."""
    match = _WORD.search(sentence)
    return match.group(0) if match else None


def _stego_gate(text: str, original: str) -> str:
    """Why an acrostic paragraph for ``original`` is rejected, or ``""``: empty,
    or the first words of its sentences are not, in order and case-folded, the
    words of the request (the exact acrostic)."""
    if not text.strip():
        return "empty"
    words = _acrostic_words(original)
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    leads = [_first_word(s) for s in sentences]
    if len(leads) != len(words):
        return f"acrostic count={len(leads)} want={len(words)}"
    for got, want in zip(leads, words):
        if got is None or got.casefold() != want.casefold():
            return "acrostic mismatch"
    return ""


# --- infoflood ------------------------------------------------------------

#: An information-overload restatement is between these multiples of the
#: original's length: long enough to be an overload, not a runaway reply.
FLOOD_RATIO_BAND = (1.8, 12.0)
#: Below this many characters a ratio says little (a short request).
MIN_FLOOD_CHARS = 20


def _infoflood_gate(text: str, original: str) -> str:
    """Why an information-overload restatement of ``original`` is rejected, or
    ``""``: empty, a copy, or a length ratio outside
    :data:`FLOOD_RATIO_BAND` (for an original of at least
    :data:`MIN_FLOOD_CHARS` characters)."""
    if not text.strip():
        return "empty"
    if text.strip() == original.strip():
        return "unchanged"
    if len(original) >= MIN_FLOOD_CHARS:
        ratio = len(text) / len(original)
        low, high = FLOOD_RATIO_BAND
        if not low <= ratio <= high:
            return f"length_ratio={ratio:.2f}"
    return ""


# --- the generate-gate-check loop -----------------------------------------


def _run(
    prompt: str,
    *,
    name: str,
    gen: list[dict],
    check,
    check_fields: Callable[[str, str], dict],
    gate: Callable[[str, str], str],
    gen_model: str | None,
    check_model: str | None,
    check_attempts: int,
) -> TechniqueGen:
    """Drive the generate-gate-check loop for a re-encoding named ``name``.

    ``gen`` is the rendered generation messages, ``check`` a loaded
    :class:`~redact.llms.prompting.PromptTemplate`, and ``check_fields(original,
    rewritten)`` the keyword arguments for it; ``gate(candidate, prompt)`` is
    the code gate. The generation runs on ``gen_model`` and the check on
    ``check_model`` (``gen_model`` when None), up to ``check_attempts``
    rewrites. Each technique builds ``gen`` and ``check`` itself, so the
    prompt steps are literal in its own code (the catalog reads them).
    """
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(gen, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(**check_fields(original, rewritten)),
        original=prompt, max_attempts=check_attempts,
        precheck=lambda candidate: gate(candidate, prompt),
        max_tokens=(GEN_MAX_TOKENS, CHECK_MAX_TOKENS),
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=reencode failed; "
                              f"form={name}; feedback={verdict}")
    return text.strip(), f"form={name}"


def _models(kwargs: dict) -> tuple:
    """``(gen_model, check_model, check_attempts, prompt_dir)`` from the chain's
    kwargs. Read here so every technique keeps the ``(prompt, **kwargs)``
    signature the registry pins, while the prompt-step literals stay in each
    technique's own code (the catalog reads them off the bytecode)."""
    return (kwargs.get("gen_model"), kwargs.get("check_model"),
            kwargs.get("check_attempts", CHECK_ATTEMPTS), kwargs.get("prompt_dir"))


def _original_rewritten(original: str, rewritten: str) -> dict:
    return {"original": original, "rewritten": rewritten}


def to_mathprompt(prompt: str, **kwargs) -> TechniqueGen:
    """Re-encode the request as a symbolic/word math problem."""
    gen_model, check_model, check_attempts, prompt_dir = _models(kwargs)
    gen = build_messages(load_prompt("jailbreak", "mathprompt", prompt_dir),
                         prompt=prompt)
    check = PromptTemplate.load("jailbreak", "mathprompt_check", prompt_dir=prompt_dir)
    return (yield from _run(
        prompt, name="math problem", gen=gen, check=check,
        check_fields=_original_rewritten, gate=_math_gate, gen_model=gen_model,
        check_model=check_model, check_attempts=check_attempts))


def to_adversarial_poetry(prompt: str, **kwargs) -> TechniqueGen:
    """Re-express the request as verse."""
    gen_model, check_model, check_attempts, prompt_dir = _models(kwargs)
    gen = build_messages(load_prompt("jailbreak", "adversarial_poetry", prompt_dir),
                         prompt=prompt)
    check = PromptTemplate.load("jailbreak", "adversarial_poetry_check",
                                prompt_dir=prompt_dir)
    return (yield from _run(
        prompt, name="poem", gen=gen, check=check,
        check_fields=_original_rewritten, gate=_poetry_gate, gen_model=gen_model,
        check_model=check_model, check_attempts=check_attempts))


@applicable_when(lambda text, entry_type: _acrostic_fits(text))
def to_stego_paragraph(prompt: str, **kwargs) -> TechniqueGen:
    """Hide the request as a word-acrostic in a benign paragraph.

    The check reads the paragraph for natural, benign prose, not for the
    request (which lives only in the acrostic the code gate verifies), so it
    is shown the hidden words and the paragraph, not the original.

    The applicability reads the plan's input, but the acrostic is built from
    the text this step is given, which an earlier step may have lengthened (a
    hacking or rewrite frame: a fresh plan keeps them apart, a kept plan or a
    manual combination may not). So the text is checked first: outside
    :data:`MIN_ACROSTIC_WORDS` to :data:`MAX_ACROSTIC_WORDS` words it is
    discarded (``reason=acrostic length``) before any model call.
    """
    if not _acrostic_fits(prompt):
        words = len(_acrostic_words(prompt))
        return prompt, (f"{DISCARDED}; reason=acrostic length; "
                        f"form=acrostic paragraph; words={words}")
    gen_model, check_model, check_attempts, prompt_dir = _models(kwargs)
    words = "\n".join(_acrostic_words(prompt))
    gen = build_messages(load_prompt("jailbreak", "stego_paragraph", prompt_dir),
                         prompt=prompt, words=words)
    check = PromptTemplate.load("jailbreak", "stego_paragraph_check",
                                prompt_dir=prompt_dir)
    return (yield from _run(
        prompt, name="acrostic paragraph", gen=gen, check=check,
        check_fields=lambda original, rewritten: {"rewritten": rewritten,
                                                  "words": words},
        gate=_stego_gate, gen_model=gen_model, check_model=check_model,
        check_attempts=check_attempts))


def to_infoflood(prompt: str, **kwargs) -> TechniqueGen:
    """Restate the request with an overload of detail and jargon."""
    gen_model, check_model, check_attempts, prompt_dir = _models(kwargs)
    gen = build_messages(load_prompt("jailbreak", "infoflood", prompt_dir),
                         prompt=prompt)
    check = PromptTemplate.load("jailbreak", "infoflood_check", prompt_dir=prompt_dir)
    return (yield from _run(
        prompt, name="information overload", gen=gen, check=check,
        check_fields=_original_rewritten, gate=_infoflood_gate, gen_model=gen_model,
        check_model=check_model, check_attempts=check_attempts))


_FNS = [to_mathprompt, to_adversarial_poetry, to_stego_paragraph, to_infoflood]


def get_semantic_reencoding_functions() -> list[Callable]:
    """Return all semantic-reencoding technique generators."""
    return list(_FNS)
