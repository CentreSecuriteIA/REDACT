"""Translation techniques: the prompt translated to one of 19 languages.

A chat translation model (the translation role's by default) translates and
checks, retrying with the check's feedback. A machine-translation model gets
the text paragraph by paragraph in its own format; each piece is checked
here (empty, language, length), then translated back and the whole judged by
the generation model, a failure resampled. ``info`` is the language, or
``DISCARDED; language=...; ...; feedback=...`` when every round is rejected.
"""

import math
import re
from collections.abc import Callable

from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import LLMRequest, is_accepted
from redact.llms.model_config import MODEL_REGISTRY, default_model_for_role
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

from . import langid

#: The registry role of a model that only translates: it takes the MT
#: prompt, and its output is checked here rather than by itself.
MACHINE_TRANSLATION_ROLE = "machine_translation"

LANGUAGES = [
    # High-resource
    "French", "Japanese", "Russian", "Spanish", "German", "Arabic",
    # Mid-resource
    "Turkish", "Czech", "Vietnamese", "Greek", "Croatian",
    # Low-resource. Not Maori: TranslateGemma invents its sensitive and
    # technical terms, fluently enough to pass every check.
    "Swahili", "Thai", "Khmer", "Nepali",
    "Zulu", "Scots Gaelic", "Bengali", "Javanese",
]

#: The ISO 639-1 code a machine-translation model is sent for each language.
#: Its template takes an unknown code as the language's name, so every
#: language needs one.
LANGUAGE_CODES = {
    "French": "fr", "Japanese": "ja", "Russian": "ru", "Spanish": "es",
    "German": "de", "Arabic": "ar", "Turkish": "tr", "Czech": "cs",
    "Vietnamese": "vi", "Greek": "el", "Croatian": "hr", "Swahili": "sw",
    "Thai": "th", "Khmer": "km", "Nepali": "ne", "Zulu": "zu",
    "Scots Gaelic": "gd", "Bengali": "bn", "Javanese": "jv",
}

#: The GlotLID labels a translation into each language may carry: its own,
#: and the near-identical languages GlotLID reads short text as.
ACCEPTED_LABELS = {
    "French": {"fra_Latn"},
    "Japanese": {"jpn_Jpan"},
    "Russian": {"rus_Cyrl"},
    "Spanish": {"spa_Latn"},
    "German": {"deu_Latn"},
    "Arabic": {"arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab",
               "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"},
    "Turkish": {"tur_Latn"},
    "Czech": {"ces_Latn"},
    "Vietnamese": {"vie_Latn"},
    "Greek": {"ell_Grek"},
    "Croatian": {"hrv_Latn", "bos_Latn", "srp_Latn"},
    "Swahili": {"swh_Latn", "swc_Latn", "swa_Latn"},
    "Thai": {"tha_Thai"},
    "Khmer": {"khm_Khmr"},
    "Nepali": {"npi_Deva"},
    "Zulu": {"zul_Latn", "nbl_Latn", "nde_Latn", "xho_Latn", "ssw_Latn"},
    "Scots Gaelic": {"gla_Latn"},
    "Bengali": {"ben_Beng"},
    "Javanese": {"jav_Latn"},  # never ind_Latn: Indonesian is a real failure
}

#: A translation's length over its input's, in characters, at the median: the
#: pod test's TranslateGemma figures, 1.0 for a language it did not measure.
MEDIAN_RATIO = {
    "French": 1.0, "Japanese": 1.0, "Russian": 1.0, "Spanish": 1.0,
    "German": 1.0, "Arabic": 1.0, "Turkish": 1.0, "Czech": 1.0,
    "Vietnamese": 1.0, "Greek": 1.0, "Croatian": 1.0, "Swahili": 1.16,
    "Thai": 1.02, "Khmer": 1.09, "Nepali": 1.04, "Zulu": 1.17,
    "Scots Gaelic": 1.32, "Bengali": 1.07, "Javanese": 1.12,
}


def _check_tables(tables: dict[str, dict]) -> None:
    """Raise ``ValueError`` unless each table names exactly :data:`LANGUAGES`."""
    for name, table in tables.items():
        missing = [language for language in LANGUAGES if language not in table]
        extra = sorted(set(table) - set(LANGUAGES))
        if missing or extra:
            raise ValueError(
                f"translation.{name} must name exactly the LANGUAGES: "
                f"missing {missing}, extra {extra}.")


_check_tables({"LANGUAGE_CODES": LANGUAGE_CODES, "ACCEPTED_LABELS": ACCEPTED_LABELS,
               "MEDIAN_RATIO": MEDIAN_RATIO})

#: A translator reply is capped at about three times its input's tokens,
#: counting a token as this many characters (which overstates an English
#: text's tokens and leaves room for a script that spends more), plus
#: :data:`REPLY_SLACK_TOKENS`, never above :data:`MAX_REPLY_TOKENS`. A reply
#: cut there is a length outlier, so a runaway loop costs one capped request.
CHARS_PER_TOKEN = 3
REPLY_TOKENS_PER_INPUT_TOKEN = 3
REPLY_SLACK_TOKENS = 32
MAX_REPLY_TOKENS = 1536


def reply_cap(text: str) -> int:
    """The ``max_tokens`` of a translator request that translates ``text``."""
    tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    return min(MAX_REPLY_TOKENS,
               REPLY_TOKENS_PER_INPUT_TOKEN * tokens + REPLY_SLACK_TOKENS)


#: The language a machine translation starts from.
SOURCE_CODE = "en"

#: A paragraph longer than this many characters is cut at sentence ends into
#: pieces of about this length at most, each its own request.
MAX_PIECE_CHARS = 1200

#: A translation's length ratio, over its language's median, must lie within.
LENGTH_RATIO_BAND = (0.5, 2.0)

#: The judge's reply cap: a verdict and a sentence.
JUDGE_MAX_TOKENS = 128

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(\s+)")


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """Where each paragraph of ``text`` is, its outer whitespace left out:
    the stretches between blank lines that hold a word."""
    spans, start = [], 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    out = []
    for start, end in spans:
        chunk = text[start:end]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            out.append((start + lead, start + lead + len(chunk.strip())))
    return out


def _sentence_pieces(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """``text[start:end]`` cut at sentence ends into pieces of at most
    :data:`MAX_PIECE_CHARS`, where its sentences allow."""
    if end - start <= MAX_PIECE_CHARS:
        return [(start, end)]
    sentences, at = [], start
    for match in _SENTENCE_END.finditer(text, start, end):
        sentences.append((at, match.start(1)))
        at = match.end()
    sentences.append((at, end))
    pieces = [sentences[0]]
    for s, e in sentences[1:]:
        if e - pieces[-1][0] <= MAX_PIECE_CHARS:
            pieces[-1] = (pieces[-1][0], e)
        else:
            pieces.append((s, e))
    return pieces


def split_text(text: str) -> tuple[list[str], list[str]]:
    """``text`` as the pieces a machine translation sends, and the gaps around
    them: ``gaps[0] + pieces[0] + gaps[1] + ... + pieces[-1] + gaps[-1]`` is
    ``text``.

    A piece is a paragraph (blank lines apart), or part of one longer than
    :data:`MAX_PIECE_CHARS`, cut at sentence ends. A text with no word is no
    piece.
    """
    spans = [piece for paragraph in _paragraphs(text)
             for piece in _sentence_pieces(text, *paragraph)]
    bounds = [0, *(at for span in spans for at in span), len(text)]
    gaps = [text[bounds[i]:bounds[i + 1]] for i in range(0, len(bounds), 2)]
    return [text[s:e] for s, e in spans], gaps


def join_pieces(pieces: list[str], gaps: list[str]) -> str:
    """The text of ``pieces`` with the ``gaps`` of :func:`split_text`."""
    return gaps[0] + "".join(p + g for p, g in zip(pieces, gaps[1:]))


def is_machine_translation(model: str) -> bool:
    """Whether ``model`` is registered with :data:`MACHINE_TRANSLATION_ROLE`;
    an unregistered name is a chat model."""
    config = MODEL_REGISTRY.get(model)
    return config is not None and MACHINE_TRANSLATION_ROLE in config.roles


def check_translation(translation: str, source: str, language: str) -> str | None:
    """Why a machine translation of ``source`` is rejected, or ``None``.

    Empty; a GlotLID label not among the language's accepted ones (skipped
    without the detector); a length ratio, over the language's median,
    outside :data:`LENGTH_RATIO_BAND`, which a reply cut at its cap is. A
    copy of ``source`` passes: the chain records it as a no-op.
    """
    if not translation.strip():
        return "empty"
    if translation == source:
        return None
    label = langid.detect(translation)
    if label is not None and label not in ACCEPTED_LABELS[language]:
        return f"language={label}"
    ratio = len(translation) / len(source) / MEDIAN_RATIO[language]
    low, high = LENGTH_RATIO_BAND
    if not low <= ratio <= high:
        return f"length_ratio={ratio:.2f}"
    return None


def _mt_messages(
    text: str, source_code: str, target_code: str, prompt_dir,
) -> list[dict]:
    """The one user message that has a machine-translation model translate
    ``text`` from ``source_code`` to ``target_code``."""
    config = load_prompt("jailbreak", "translate_mt", prompt_dir)
    return build_messages(config, source_code=source_code, target_code=target_code,
                          text=text)


#: The markers of the machine-translation format, in their order.
_MT_MARKERS = ("<<<source>>>", "<<<target>>>", "<<<text>>>")


def check_mt_prompt(prompt_dir=None) -> None:
    """Raise unless ``jailbreak/translate_mt`` renders to one user message
    holding the model's three markers in order: its chat template raises on
    anything else, which would fail every translation.

    Raises:
        ValueError: The prompt renders otherwise; the message names its file.
    """
    sample = {"source_code": SOURCE_CODE, "target_code": "sw", "text": "text"}
    template = PromptTemplate.load("jailbreak", "translate_mt",
                                   prompt_dir=prompt_dir, **sample)
    messages = template(**sample)
    roles = [m["role"] for m in messages]
    content = messages[0]["content"] if roles == ["user"] else ""
    at = [content.find(marker) for marker in _MT_MARKERS]
    if -1 in at or at != sorted(at):
        origin = "override" if template.is_override else "packaged"
        raise ValueError(
            f"The machine-translation prompt {template.source} ({origin}) must "
            f"render to one user message holding {', '.join(_MT_MARKERS)} in this "
            f"order; it renders to {roles}. The model's "
            f"chat template refuses anything else.")


def _judge_messages(original: str, back_translation: str, language: str,
                    prompt_dir) -> list[dict]:
    """The judge's messages: the original against its back-translation."""
    config = load_prompt("jailbreak", "translate_check_back", prompt_dir)
    return build_messages(config, original=original,
                          back_translation=back_translation, language=language)


def _machine_translate_gen(
    prompt: str, language: str, *, translator: str, judge: str, num_retries: int,
    prompt_dir,
) -> TechniqueGen:
    """Translate ``prompt`` piece by piece on ``translator``, one request at a
    time.

    A round translates the pieces still pending and checks each
    (:func:`check_translation`); a piece that fails is resampled alone in the
    next round. When every piece passes, each is translated back and
    ``judge`` reads the original against the joined back-translation; a
    rejection resamples every piece in the next round. A resample is the same
    request again: the model's temperature varies it. After ``num_retries``
    rounds the info names the piece that failed (``piece=all`` for the judge)
    and why.
    """
    pieces, gaps = split_text(prompt)
    if not pieces:
        return prompt, language
    code = LANGUAGE_CODES[language]
    translations = list(pieces)
    pending = list(range(len(pieces)))
    reason = ""
    for _ in range(num_retries):
        failed = []
        for i in pending:
            translations[i] = yield LLMRequest(
                translator, _mt_messages(pieces[i], SOURCE_CODE, code, prompt_dir),
                max_tokens=reply_cap(pieces[i]))
            why = check_translation(translations[i], pieces[i], language)
            if why is not None:
                failed.append((i, why))
        if failed:
            first, why = failed[0]
            reason = f"piece={first + 1}/{len(pieces)}; feedback={why}"
            pending = [i for i, _ in failed]
            continue
        backs = []
        for translated in translations:
            backs.append((yield LLMRequest(
                translator, _mt_messages(translated, code, SOURCE_CODE, prompt_dir),
                max_tokens=reply_cap(translated))))
        verdict = yield LLMRequest(
            judge, _judge_messages(prompt, join_pieces(backs, gaps), language,
                                   prompt_dir),
            max_tokens=JUDGE_MAX_TOKENS)
        if is_accepted(verdict):
            return join_pieces(translations, gaps), language
        reason = f"piece=all; feedback=judge={verdict}"
        pending = list(range(len(pieces)))
    return (join_pieces(translations, gaps),
            f"{DISCARDED}; language={language}; {reason}")


def _resolve_translate_model() -> str:
    """Return the canonical translation model from the registry.

    Looked up by role so the choice can be changed by registering a
    different model with ``role="translation"``. Falls back to the packaged
    default ``translategemma-27b`` if the role isn't registered (e.g. user
    mutated the registry).
    """
    try:
        return default_model_for_role("translation")
    except KeyError:
        return "translategemma-27b"


def _build_translate_messages(
    text: str, language: str, feedback: str = "", prompt_dir=None,
) -> list[dict]:
    """Build the chat messages for a translation request (pure, no LLM call).

    Loads ``translate`` (first attempt) or ``translate_retry`` (feedback
    from a prior rejected attempt) — two separate prompt categories rather
    than one template with a conditional block, since ``str.format_map()``
    has no conditionals.
    """
    category = "translate_retry" if feedback else "translate"
    config = load_prompt("jailbreak", category, prompt_dir)
    return build_messages(config, language=language, text=text, feedback=feedback)


def _build_check_messages(
    original: str, translation: str, language: str, prompt_dir=None,
) -> list[dict]:
    """Build the chat messages for a translation-quality check (pure)."""
    config = load_prompt("jailbreak", "translate_check", prompt_dir)
    return build_messages(config, original=original, translation=translation, language=language)


def _translate_gen(
    prompt: str,
    language: str,
    *,
    translate_model: str | None = None,
    check_model: str | None = None,
    gen_model: str | None = None,
    num_retries: int = 4,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Translate ``prompt`` to ``language``, in up to ``num_retries`` rounds.

    ``translate_model`` translates, the translation role's model when None.
    A machine-translation model (:func:`is_machine_translation`) takes
    :func:`_machine_translate_gen`'s path, judged by ``check_model``, else
    ``gen_model`` (the stage's generation model), else the ``uncensored_gen``
    default. A chat model translates and checks each round, ``check_model``
    replacing it for the check, the check's verdict fed back to the next
    round. Returns ``(translation, language)``, or ``(last_attempt,
    "DISCARDED; language=...; ...")`` on exhaustion.

    ``prompt_dir`` is the overlay the prompts are read through. ``**kwargs``
    swallows engine-supplied keys (``benign_data``, ``rng``) that translation
    does not use.
    """
    # Resolved fresh per call (not a frozen import-time constant) so a
    # runtime register_model(..., role="translation") takes effect
    # immediately, same as every other role default in the library.
    translator = translate_model or _resolve_translate_model()
    if is_machine_translation(translator):
        judge = check_model or gen_model or default_model_for_role("uncensored_gen")
        return (yield from _machine_translate_gen(
            prompt, language, translator=translator, judge=judge,
            num_retries=num_retries, prompt_dir=prompt_dir))
    chk_model = check_model or translator
    feedback = ""
    last = prompt
    for _ in range(num_retries):
        translation = yield LLMRequest(
            translator,
            _build_translate_messages(prompt, language, feedback, prompt_dir),
        )
        verdict = yield LLMRequest(
            chk_model, _build_check_messages(prompt, translation, language, prompt_dir)
        )
        if is_accepted(verdict):
            return translation, language
        feedback = verdict
        last = translation
    return last, f"{DISCARDED}; language={language}; feedback={feedback}"


def _make_language_fn(language: str) -> Callable:
    """The technique that translates to ``language``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _translate_gen(prompt, language, **kwargs))

    fn.__name__ = fn.__qualname__ = "to_" + language.lower().replace(" ", "_")
    fn.__doc__ = f"Translate to {language} (technique generator)."
    return fn


# One technique per language, named to_<language>: to_french, to_scots_gaelic, ...
_FNS = [_make_language_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_translation_functions() -> list[Callable]:
    """Return all translation technique generators."""
    return list(_FNS)
