"""Translationese: the prompt round-tripped through a pivot language on the
translation model (English to the pivot and back), so the English reads as
machine-translated (spec 1.5, adversarial only).

One technique, :func:`to_translationese`. It reuses the translation machinery
(``translation.py``) and keeps translation's own checks (``check`` ``translate``):
a machine-translation model translates each piece to the pivot and back, each
leg checked by :func:`~redact.jailbreak.obfuscation.translation.check_translation`
(the pivot's GlotLID labels and length band on the way out, English and a
length band on the way back), the English output must differ from the original,
and the generation model judges the original against the joined output; a chat
translator does the two legs by name. The pivot is drawn from
:data:`PIVOTS` with the unit's rng. ``info`` is ``pivot=<name>``;
``DISCARDED; language=translationese; pivot=<name>; ...`` on failure.
"""

import random
from collections.abc import Callable

from redact.jailbreak.technique import DISCARDED, TechniqueGen, is_discarded
from redact.llm_pipeline import LLMRequest, is_accepted
from redact.llms.model_config import default_model_for_role

from .translation import (
    ACCEPTED_LABELS,
    JUDGE_MAX_TOKENS,
    LANGUAGE_CODES,
    MEDIAN_RATIO,
    SOURCE_CODE,
    _chat_translate_gen,
    _judge_messages,
    _mt_messages,
    _resolve_translate_model,
    check_translation,
    is_machine_translation,
    join_pieces,
    reply_cap,
    split_text,
)

#: The pivot languages a round trip may use, each a shipped translation
#: language (its code, labels and median ratio are in ``translation.py``). High
#: and mid-resource languages, so the forward translation is reliable and the
#: round trip reads as translationese rather than breaking the meaning.
PIVOTS = ("french", "german", "russian", "spanish", "chinese")

#: The English round trip must differ from the original and stay in this length
#: band (over a ratio of 1.0, English to English).
ENGLISH_LABELS = {"eng_Latn"}
ENGLISH_RATIO = 1.0


def _pivot(kwargs: dict) -> str:
    """The pivot language for this unit: drawn from :data:`PIVOTS` with the
    chain's ``rng`` when it has one, else at random (a direct call)."""
    rng = kwargs.get("rng")
    return (rng or random).choice(PIVOTS)


def _round_trip_mt(
    prompt: str, *, pivot: str, translator: str, judge: str, num_retries: int,
    prompt_dir,
) -> TechniqueGen:
    """Round-trip ``prompt`` through ``pivot`` on a machine translator, piece by
    piece: translate to the pivot, back to English, keep the English. A piece
    whose forward or back leg fails a check, or whose round trip is unchanged,
    is resampled; when every piece passes, ``judge`` reads the original against
    the joined English round trip. After ``num_retries`` rounds the info names
    the piece and why."""
    code = LANGUAGE_CODES[pivot]
    labels, ratio = ACCEPTED_LABELS[pivot], MEDIAN_RATIO[pivot]
    pieces, gaps = split_text(prompt)
    if not pieces:
        return prompt, f"pivot={pivot}"
    english = list(pieces)
    pending = list(range(len(pieces)))
    reason = ""
    for _ in range(num_retries):
        failed = []
        for i in pending:
            forward = yield LLMRequest(
                translator, _mt_messages(pieces[i], SOURCE_CODE, code, prompt_dir),
                max_tokens=reply_cap(pieces[i]))
            why = check_translation(forward, pieces[i], labels=labels,
                                    median_ratio=ratio)
            if why is not None:
                failed.append((i, f"forward:{why}"))
                continue
            back = yield LLMRequest(
                translator, _mt_messages(forward, code, SOURCE_CODE, prompt_dir),
                max_tokens=reply_cap(forward))
            why = check_translation(back, pieces[i], labels=ENGLISH_LABELS,
                                    median_ratio=ENGLISH_RATIO)
            if why is not None:
                failed.append((i, f"back:{why}"))
                continue
            if back.strip() == pieces[i].strip():
                failed.append((i, "unchanged"))
                continue
            english[i] = back
        if failed:
            first, why = failed[0]
            reason = f"piece={first + 1}/{len(pieces)}; feedback={why}"
            pending = [i for i, _ in failed]
            continue
        verdict = yield LLMRequest(
            judge, _judge_messages(prompt, join_pieces(english, gaps), "English",
                                   prompt_dir),
            max_tokens=JUDGE_MAX_TOKENS)
        if is_accepted(verdict):
            return join_pieces(english, gaps), f"pivot={pivot}"
        reason = f"piece=all; feedback=judge={verdict}"
        pending = list(range(len(pieces)))
    return (join_pieces(english, gaps),
            f"{DISCARDED}; language=translationese; pivot={pivot}; {reason}")


def _round_trip_chat(
    prompt: str, *, pivot: str, translator: str, checker: str, num_retries: int,
    prompt_dir,
) -> TechniqueGen:
    """Round-trip ``prompt`` through ``pivot`` on a chat translator: translate
    to the pivot by name, back to English, each leg checked by ``checker`` (the
    translation family's own meaning check). The English round trip must differ
    from the original."""
    pivot_text, info = yield from _chat_translate_gen(
        prompt, pivot, translator=translator, checker=checker,
        num_retries=num_retries, prompt_dir=prompt_dir)
    if is_discarded(info):
        return pivot_text, f"{DISCARDED}; language=translationese; pivot={pivot}; " \
                           f"feedback=to pivot: {info[len(f'{DISCARDED}; '):]}"
    english, info = yield from _chat_translate_gen(
        pivot_text, "English", translator=translator, checker=checker,
        num_retries=num_retries, prompt_dir=prompt_dir)
    if is_discarded(info):
        return english, f"{DISCARDED}; language=translationese; pivot={pivot}; " \
                        f"feedback=to english: {info[len(f'{DISCARDED}; '):]}"
    if english.strip() == prompt.strip():
        return prompt, (f"{DISCARDED}; language=translationese; pivot={pivot}; "
                        "feedback=unchanged")
    return english, f"pivot={pivot}"


def to_translationese(prompt: str, **kwargs) -> TechniqueGen:
    """Round-trip the prompt through a pivot language so it reads as translated.

    Runs on the translation model (``translate_model``, the translation role's
    by default). A machine-translation model takes the piece-by-piece path
    (judged by ``translate_check_model``, else ``gen_model``, else the
    ``uncensored_gen`` default); a chat model does the two legs by name.
    """
    translate_model = kwargs.get("translate_model")
    translate_check_model = kwargs.get("translate_check_model")
    gen_model = kwargs.get("gen_model")
    num_retries = kwargs.get("num_retries", 4)
    prompt_dir = kwargs.get("prompt_dir")
    pivot = _pivot(kwargs)
    translator = translate_model or _resolve_translate_model()
    if is_machine_translation(translator):
        judge = (translate_check_model or gen_model
                 or default_model_for_role("uncensored_gen"))
        return (yield from _round_trip_mt(
            prompt, pivot=pivot, translator=translator, judge=judge,
            num_retries=num_retries, prompt_dir=prompt_dir))
    return (yield from _round_trip_chat(
        prompt, pivot=pivot, translator=translator,
        checker=translate_check_model or translator, num_retries=num_retries,
        prompt_dir=prompt_dir))


_FNS: list[Callable] = [to_translationese]


def get_translationese_functions() -> list[Callable]:
    """Return the translationese technique generator."""
    return list(_FNS)
