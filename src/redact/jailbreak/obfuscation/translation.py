"""Translation techniques: the prompt translated to one of the languages of
``configs/jailbreak/languages.json`` (19 today).

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

from redact.jailbreak.technique import DISCARDED, TechniqueGen, data_rows
from redact.llm_pipeline import LLMRequest, is_accepted
from redact.llms.model_config import MODEL_REGISTRY, default_model_for_role
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

from . import langid

#: The registry role of a model that only translates: it takes the MT
#: prompt, and its output is checked here rather than by itself.
MACHINE_TRANSLATION_ROLE = "machine_translation"

#: The languages a prompt is translated to: the rows of
#: ``configs/jailbreak/languages.json`` whose ``translation`` is true, in its
#: order, each with its ``name``, ``slug`` (``to_<slug>``), ``complexity``
#: and ``script`` (the ISO 15924 code of its script, the suffix of its
#: GlotLID labels; the rules keep a translation into a script other than
#: :data:`~redact.jailbreak.technique.LATIN` from a ``needs_latin``
#: technique). The tables below are keyed by the slug.
LANGUAGES = [row for row in data_rows("languages.json") if row["translation"]]

#: The ISO 639-1 code a machine-translation model is sent for each language.
#: Its template takes an unknown code as the language's name, so every
#: language needs one.
LANGUAGE_CODES = {
    "french": "fr", "japanese": "ja", "russian": "ru", "spanish": "es",
    "german": "de", "arabic": "ar", "turkish": "tr", "czech": "cs",
    "vietnamese": "vi", "greek": "el", "croatian": "hr", "swahili": "sw",
    "thai": "th", "khmer": "km", "nepali": "ne", "zulu": "zu",
    "scots_gaelic": "gd", "bengali": "bn", "javanese": "jv",
}

#: The GlotLID labels a translation into each language may carry: its own,
#: and the near-identical languages GlotLID reads short text as.
ACCEPTED_LABELS = {
    "french": {"fra_Latn"},
    "japanese": {"jpn_Jpan"},
    "russian": {"rus_Cyrl"},
    "spanish": {"spa_Latn"},
    "german": {"deu_Latn"},
    "arabic": {"arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab",
               "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"},
    "turkish": {"tur_Latn"},
    "czech": {"ces_Latn"},
    "vietnamese": {"vie_Latn"},
    "greek": {"ell_Grek"},
    "croatian": {"hrv_Latn", "bos_Latn", "srp_Latn"},
    "swahili": {"swh_Latn", "swc_Latn", "swa_Latn"},
    "thai": {"tha_Thai"},
    "khmer": {"khm_Khmr"},
    "nepali": {"npi_Deva"},
    "zulu": {"zul_Latn", "nbl_Latn", "nde_Latn", "xho_Latn", "ssw_Latn"},
    "scots_gaelic": {"gla_Latn"},
    "bengali": {"ben_Beng"},
    "javanese": {"jav_Latn"},  # never ind_Latn: Indonesian is a real failure
}

#: A translation's length over its input's, in characters, at the median,
#: measured: TranslateGemma 27B translating the FLORES-200 devtest (1,012
#: sentences) from English, on 2026-10-19. The seven figures kept from an
#: earlier pod test (Swahili to Javanese, Scots Gaelic aside) agree with that
#: run within 0.02.
MEDIAN_RATIO = {
    "french": 1.27, "japanese": 0.47, "russian": 1.18, "spanish": 1.22,
    "german": 1.24, "arabic": 0.96, "turkish": 1.13, "czech": 1.05,
    "vietnamese": 1.16, "greek": 1.28, "croatian": 1.07, "swahili": 1.16,
    "thai": 1.02, "khmer": 1.09, "nepali": 1.04, "zulu": 1.17,
    "scots_gaelic": 1.31, "bengali": 1.07, "javanese": 1.12,
}


def _check_tables(tables: dict[str, dict]) -> None:
    """Raise ``ValueError`` unless each table names exactly the slugs of
    :data:`LANGUAGES`: a language added to ``languages.json`` for
    translation needs its code, labels and measured median here."""
    slugs = [language["slug"] for language in LANGUAGES]
    for name, table in tables.items():
        missing = [slug for slug in slugs if slug not in table]
        extra = sorted(set(table) - set(slugs))
        if missing or extra:
            raise ValueError(
                f"translation.{name} must name exactly the translation languages "
                f"of languages.json, by slug: missing {missing}, extra {extra}.")


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


def check_translation(translation: str, source: str, slug: str) -> str | None:
    """Why a machine translation of ``source`` into the language ``slug``
    is rejected, or ``None``.

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
    if label is not None and label not in ACCEPTED_LABELS[slug]:
        return f"language={label}"
    ratio = len(translation) / len(source) / MEDIAN_RATIO[slug]
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
    prompt: str, language: dict, *, translator: str, judge: str, num_retries: int,
    prompt_dir,
) -> TechniqueGen:
    """Translate ``prompt`` piece by piece on ``translator``, one request at a
    time, into ``language`` (a row of :data:`LANGUAGES`).

    A round translates the pieces still pending and checks each
    (:func:`check_translation`); a piece that fails is resampled alone in the
    next round. When every piece passes, each is translated back and
    ``judge`` reads the original against the joined back-translation; a
    rejection resamples every piece in the next round. A resample is the same
    request again: the model's temperature varies it. After ``num_retries``
    rounds the info names the piece that failed (``piece=all`` for the judge)
    and why.
    """
    name, slug = language["name"], language["slug"]
    pieces, gaps = split_text(prompt)
    if not pieces:
        return prompt, name
    code = LANGUAGE_CODES[slug]
    translations = list(pieces)
    pending = list(range(len(pieces)))
    reason = ""
    for _ in range(num_retries):
        failed = []
        for i in pending:
            translations[i] = yield LLMRequest(
                translator, _mt_messages(pieces[i], SOURCE_CODE, code, prompt_dir),
                max_tokens=reply_cap(pieces[i]))
            why = check_translation(translations[i], pieces[i], slug)
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
            judge, _judge_messages(prompt, join_pieces(backs, gaps), name,
                                   prompt_dir),
            max_tokens=JUDGE_MAX_TOKENS)
        if is_accepted(verdict):
            return join_pieces(translations, gaps), name
        reason = f"piece=all; feedback=judge={verdict}"
        pending = list(range(len(pieces)))
    return (join_pieces(translations, gaps),
            f"{DISCARDED}; language={name}; {reason}")


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
    language: dict,
    *,
    translate_model: str | None = None,
    check_model: str | None = None,
    gen_model: str | None = None,
    num_retries: int = 4,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Translate ``prompt`` to ``language`` (a row of :data:`LANGUAGES`), in
    up to ``num_retries`` rounds.

    ``translate_model`` translates, the translation role's model when None.
    A machine-translation model (:func:`is_machine_translation`) takes
    :func:`_machine_translate_gen`'s path, judged by ``check_model``, else
    ``gen_model`` (the stage's generation model), else the ``uncensored_gen``
    default. A chat model translates and checks each round, ``check_model``
    replacing it for the check, the check's verdict fed back to the next
    round. Returns ``(translation, name)``, the language's name, or
    ``(last_attempt, "DISCARDED; language=<name>; ...")`` on exhaustion.

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
    name = language["name"]
    chk_model = check_model or translator
    feedback = ""
    last = prompt
    for _ in range(num_retries):
        translation = yield LLMRequest(
            translator,
            _build_translate_messages(prompt, name, feedback, prompt_dir),
        )
        verdict = yield LLMRequest(
            chk_model, _build_check_messages(prompt, translation, name, prompt_dir)
        )
        if is_accepted(verdict):
            return translation, name
        feedback = verdict
        last = translation
    return last, f"{DISCARDED}; language={name}; feedback={feedback}"


def _make_language_fn(language: dict) -> Callable:
    """The technique that translates to ``language`` (a row of
    :data:`LANGUAGES`), named ``to_<slug>``, with the language's ``script``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _translate_gen(prompt, language, **kwargs))

    fn.__name__ = fn.__qualname__ = f"to_{language['slug']}"
    fn.__doc__ = f"Translate to {language['name']} (technique generator)."
    fn.script = language["script"]
    return fn


# One technique per language, named to_<slug>: to_french, to_scots_gaelic, ...
_FNS = [_make_language_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_translation_functions() -> list[Callable]:
    """Return all translation technique generators."""
    return list(_FNS)
