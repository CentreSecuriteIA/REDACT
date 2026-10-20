"""The paraphrase stage: K checked paraphrases per row, over a pool of paraphrasers.

Any capable model of the ``paraphraser`` role stands in for the trained
defingerprinting model, which lives in another repository.

Each paraphraser's units run on the ``drive_units`` pool, one paraphraser
at a time, each unit with its own reply cap (:func:`reply_cap`). Each
paraphrase is checked in two steps. Five deterministic gates
(:data:`GATES`: length, split lines, script, language, artifacts) reject
what they can without a model call; the rest go to a separate model at
``llm_pipeline.CHECK_TEMPERATURE`` (0.15), with the strict prompt of
:func:`~redact.content_moderation.checker.build_paraphrase_checker`. The
language gate reads GlotLID from ``jailbreak/obfuscation/langid.py``, the
translation check's identifier, imported where it is used.
"""

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from itertools import repeat
from pathlib import Path

import pandas as pd

from ..dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    commit,
    content_id,
    count_rows,
    read_rows,
    release_lock,
    resume_state,
    row_ids,
)
from ..exceptions import PromptTooLongError
from ..llm_pipeline import CHECK_TEMPERATURE, checked, drive_units, pool_size
from ..llms.backends import apply_system_prompt_policy
from ..llms.client import ModelClient
from ..llms.model_config import (
    MODEL_REGISTRY,
    default_model_for_role,
    get_models_by_role,
)
from ..llms.prompting import PromptTemplate, load_prompt
from ..llms.router import batch_generate_samples
from .checker import build_paraphrase_checker

logger = logging.getLogger(__name__)

# The columns a paraphrase of a response takes from it: the response's id
# (its unit's key) and how the response answers its input.
_RESPONSE_COLUMNS = ("response_id", "response_mode", "output_label", "response_label")
_LABEL_COLUMNS = _RESPONSE_COLUMNS[1:]

#: What a call or a recipe that names no ``target`` paraphrases: the inputs.
#: The outputs target (``"outputs"``, or ``"both"`` with the inputs) is an
#: opt-in.
DEFAULT_TARGET = "inputs"

#: A text under this many words gets no paraphrase. Below 12 words the
#: humanizer read a text as a headline and wrote a post under it (net
#: acceptance 0 to 50% by word count on the benign test, 2026-10-20, against
#: 91.7% at 12 to 15 words); every test input under 12 words was a fragment.
MIN_WORDS = 12


def skip_reason(text: str, *, min_words: int) -> str:
    """Why ``text`` gets no paraphrase: ``"short"`` when it has under
    ``min_words`` words (split on whitespace), else ``""``."""
    return "short" if len(text.split()) < min_words else ""


# --- the check's gates -------------------------------------------------------
#
# Each gate is ``(original, paraphrase) -> reason``, ``""`` when it passes, and
# strips both texts itself. A gated paraphrase is saved as a rejected row and
# is not sent to the check model. The thresholds were measured on 1,638 benign
# rewrites of REDACT texts by two paraphrasers (2026-10-20), set so that no
# rewrite read as good fails a gate.

#: The squeezed paraphrase may be at most ``LENGTH_HIGH`` times the squeezed
#: original plus ``LENGTH_SLACK`` characters, and at least ``LENGTH_LOW``
#: times it. The nearest good rewrites were 1.40x (at +20) and 0.516x.
LENGTH_HIGH = 1.5
LENGTH_SLACK = 20
LENGTH_LOW = 0.5

#: A single-line original of at most this many words must stay on one line.
#: The smallest good multi-line rewrite of a single-line original had 43.
SPLIT_WORDS = 35

#: A script that makes up under ``SCRIPT_RARE`` of the original's letters
#: counts as absent from it; the gate fires when ``SCRIPT_SHARE`` or more of
#: the paraphrase's letters are in such scripts. The nearest good rewrite had
#: 0.07% of its letters in them.
SCRIPT_SHARE = 0.10
SCRIPT_RARE = 0.01

#: GlotLID labels the two texts differently with at least this probability
#: on both. The nearest good rewrite had 0.671; below 0.7 GlotLID labels
#: code and tables as a language.
LANGID_PROBABILITY = 0.9

_WORD = re.compile(r"[\w']+")
_SCRIPTS = ("LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC",
            "ARABIC", "DEVANAGARI", "THAI", "GREEK", "HEBREW")
# An HTML tag, or a rule line, that a model's web text leaves in a reply.
_ARTIFACT = re.compile(
    r"</?(?:p|br|div|span|b|i|ul|li|h[1-6])\b[^>]*>|^\s*(?:={4,}|-{6,}|_{6,})\s*$",
    re.IGNORECASE | re.MULTILINE)


def squeezed(text: str) -> str:
    """``text`` with its layout taken out and its content kept: table rule
    lines and lines of only ``-=_*|:`` become a space, every ``|`` a space,
    and each run of whitespace one space."""
    text = re.sub(r"^\s*\|?\s*:?-{2,}.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s\-=_*|:]+$", " ", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    return " ".join(text.split())


def length_gate(original: str, paraphrase: str) -> str:
    """Fires when the squeezed paraphrase is over ``LENGTH_HIGH`` times the
    squeezed original plus ``LENGTH_SLACK`` characters, or under
    ``LENGTH_LOW`` times it: a run-on, an answer, or a reply cut short."""
    o, p = len(squeezed(original)), len(squeezed(paraphrase))
    if not o:
        return f"length {p} against an empty original" if p > LENGTH_SLACK else ""
    if p > LENGTH_HIGH * o + LENGTH_SLACK or p < LENGTH_LOW * o:
        return f"length ratio {p / o:.2f}"
    return ""


def split_gate(original: str, paraphrase: str) -> str:
    """Fires when a single-line original of at most ``SPLIT_WORDS`` words
    comes back on two or more non-empty lines."""
    if "\n" in original.strip() or len(_WORD.findall(original)) > SPLIT_WORDS:
        return ""
    lines = [line for line in paraphrase.strip().split("\n") if line.strip()]
    return "split into lines" if len(lines) > 1 else ""


def _script_counts(text: str) -> Counter:
    """The letters of ``text`` per script, read from each character's
    Unicode name; digits, punctuation and emoji count for none."""
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        script = next((s for s in _SCRIPTS if name.startswith(s)), None)
        counts["Han" if script == "CJK" else script.title() if script else "Other"] += 1
    return counts


def script_gate(original: str, paraphrase: str) -> str:
    """Fires when ``SCRIPT_SHARE`` or more of the paraphrase's letters are in
    scripts that make up under ``SCRIPT_RARE`` of the original's letters
    (English answered in Chinese). A text with no letters never fires it.
    The reason names the original's main script and the main one of those
    the original lacks."""
    ours = _script_counts(original.strip())
    theirs = _script_counts(paraphrase.strip())
    n_ours, n_theirs = sum(ours.values()), sum(theirs.values())
    if not n_ours or not n_theirs:
        return ""
    foreign = Counter({script: n for script, n in theirs.items()
                       if ours.get(script, 0) / n_ours < SCRIPT_RARE})
    if sum(foreign.values()) / n_theirs < SCRIPT_SHARE:
        return ""
    return (f"script changed: {ours.most_common(1)[0][0]} -> "
            f"{foreign.most_common(1)[0][0]}")


def language_gate(original: str, paraphrase: str) -> str:
    """Fires when GlotLID labels the two texts differently, each with a
    probability of ``LANGID_PROBABILITY`` or above. Without the ``langid``
    extra or its model it passes every text (the front warns once)."""
    from ..jailbreak.obfuscation import langid

    ours = langid.label_probability(original.strip())
    theirs = langid.label_probability(paraphrase.strip())
    if ours is None or theirs is None:
        return ""
    (a, pa), (b, pb) = ours, theirs
    if a != b and pa >= LANGID_PROBABILITY and pb >= LANGID_PROBABILITY:
        return f"language {a} -> {b}"
    return ""


def artifact_gate(original: str, paraphrase: str) -> str:
    """Fires when the paraphrase holds an HTML tag or a rule line and the
    original holds none at all."""
    found = _ARTIFACT.search(paraphrase.strip())
    if found is None or _ARTIFACT.search(original.strip()):
        return ""
    return f"artifact {found.group().strip()}"


#: The gates, in the order they are tried.
GATES = (length_gate, split_gate, script_gate, language_gate, artifact_gate)


def gate_reason(original: str, paraphrase: str) -> str:
    """The reason of the first gate in :data:`GATES` that ``paraphrase``
    fails, or ``""`` when it passes them all."""
    for gate in GATES:
        reason = gate(original, paraphrase)
        if reason:
            return reason
    return ""


def paraphrase_prompt(
    model: str, prompt_dir: str | Path | None = None,
) -> PromptTemplate:
    """The paraphrase prompt of ``model``: ``paraphrase/generation/<model>``
    when it exists, else the generic ``paraphrase/generation``.

    The directory is the model's registry name. A prompt is looked for in
    this order: the override's per-model file, the packaged per-model file,
    the override's generic file, the packaged generic file. So a generic
    override does not reach a model that has a per-model prompt.
    """
    try:
        return PromptTemplate.load("paraphrase", f"generation/{model}",
                                   prompt_dir=prompt_dir)
    except FileNotFoundError:
        return PromptTemplate.load("paraphrase", "generation", prompt_dir=prompt_dir)


#: A paraphrase's reply cap: REPLY_TOKENS_PER_INPUT_TOKEN times the
#: original's tokens, estimated at CHARS_PER_TOKEN characters a token, plus
#: REPLY_SLACK_TOKENS, at least REPLY_FLOOR_TOKENS. The humanizer's card
#: allows "about 2.5 times the draft's token count"; the floor keeps a short
#: text from running to the model's whole default (a 5-word fragment ran to
#: its 256-token cap in the benign test). The same shape as the translation
#: technique's reply_cap (jailbreak/obfuscation/translation.py), with other
#: figures.
REPLY_TOKENS_PER_INPUT_TOKEN = 2.5
REPLY_FLOOR_TOKENS = 64
REPLY_SLACK_TOKENS = 32
CHARS_PER_TOKEN = 3


def reply_cap(text: str, ceiling: int) -> int:
    """The reply cap of a paraphrase of ``text``, at most ``ceiling`` (the
    paraphraser's ``default_max_tokens``). A reply cut at it fails the length
    gate."""
    tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    return min(ceiling, max(REPLY_FLOOR_TOKENS,
                            math.ceil(REPLY_TOKENS_PER_INPUT_TOKEN * tokens)
                            + REPLY_SLACK_TOKENS))


def _unit_gen(client, check, checker, template, original, *, provisional_id=None):
    """One unit: a paraphrase of ``original``, capped at its reply cap, then
    the gates and the check (:func:`~redact.llm_pipeline.checked`, one
    attempt). Without ``check`` the reply is accepted and no gate runs. The
    check sees the paraphrase stripped, as the gates do."""
    return checked(
        client, lambda feedback: template(sample=original.strip()),
        check,
        (lambda o, text: checker(o, text.strip())) if check is not None else None,
        original=original,
        max_tokens=(reply_cap(original, client.backend.default_max_tokens), None),
        precheck=lambda text: gate_reason(original, text),
        temperatures=(None, CHECK_TEMPERATURE),
        internals_ids=(provisional_id, None),
        max_attempts=1,
    )


def check_paraphrase_prompt(model: str, prompt_dir: str | Path | None = None) -> None:
    """Check, before any call, that ``model``'s paraphrase prompt renders the
    format its ``fingerprint`` key pins.

    A prompt file may hold ``"fingerprint": {"sample": ..., "sha256_16":
    ...}``: the prompt rendered for ``sample``, under the model's
    system-prompt policy, has a last user message whose SHA-256 begins with
    ``sha256_16`` (16 hex characters). A model that formats its own prompt
    (the humanizer, a text-completion model) gives junk on any other format,
    so a broken override stops the stage here, as ``check_mt_prompt`` stops
    the jailbreak stage. A file without the key is not checked; when it is
    an override of a packaged prompt that has one, a WARNING says its format
    is not checked. An unregistered ``model`` is left to
    ``ModelClient.create``, which names it.

    Raises:
        ValueError: The key is malformed, a model without a system role gets
            more than one user message, or the hash differs. The message
            names the file, its origin and both hashes.
    """
    config = MODEL_REGISTRY.get(model)
    if config is None:
        return
    template = paraphrase_prompt(model, prompt_dir)
    where = f"{template.source} ({'override' if template.is_override else 'packaged'})"
    pipeline, category = template.name.split("/", 1)
    key = load_prompt(pipeline, category, prompt_dir).get("fingerprint")
    if key is None:
        if template.is_override:
            try:
                packaged = load_prompt("paraphrase", f"generation/{model}")
            except FileNotFoundError:
                packaged = {}
            if "fingerprint" in packaged:
                logger.warning(
                    "paraphrase: %s, the paraphrase prompt of %s, has no fingerprint "
                    "key, so its format is not checked.", where, model)
        return
    if not (isinstance(key, dict) and isinstance(key.get("sample"), str)
            and isinstance(key.get("sha256_16"), str)):
        raise ValueError(
            f"The fingerprint of {where} must be "
            f'{{"sample": "<text>", "sha256_16": "<16 hex>"}}, got {key!r}.')
    _system, chat = apply_system_prompt_policy(
        template(sample=key["sample"]), config.supports_system_prompt)
    if not config.supports_system_prompt and [m.get("role") for m in chat] != ["user"]:
        raise ValueError(
            f"{where} renders {len(chat)} messages for {model}, which takes "
            "exactly one user message (it has no system role); its few-shot "
            "examples or roles do not fit the model's format.")
    users = [m["content"] for m in chat if m.get("role") == "user"]
    content = users[-1] if users else ""
    found = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    if found != key["sha256_16"]:
        raise ValueError(
            f"The paraphrase prompt of {model}, {where}, does not render the "
            f"format its fingerprint pins: for the sample {key['sample']!r} its "
            f"prompt hashes to {found}, not {key['sha256_16']}. Restore the "
            "file, or remove its fingerprint key to run another wording "
            "unchecked.")


def paraphrase_batch(
    client: ModelClient,
    samples: list[str],
    prompt_dir: str | None = None,
    progress: str | None = None,
    **kwargs,
) -> list[str]:
    """Paraphrase each sample, in order, in one batch of calls.

    Each sample is stripped and rendered with the client's own prompt
    (:func:`paraphrase_prompt`). ``progress`` labels the batch; ``kwargs``
    go to ``batch_generate_samples``. A helper for callers: the stage no
    longer calls it, since its units run on the ``drive_units`` pool.
    """
    if not samples:
        return []
    template = paraphrase_prompt(client.model, prompt_dir)
    return batch_generate_samples(
        client, [template(sample=s.strip()) for s in samples],
        batch_size=len(samples), progress=progress, **kwargs,
    )


def paraphrase_pool(paraphraser: str | None) -> list[str]:
    """Resolve the paraphraser model pool.

    ``None`` → the ``paraphraser`` role default; a model name → just that model;
    ``"distribution"`` → all models registered with ``role="paraphraser"``.
    """
    if paraphraser == "distribution":
        seen: set[str] = set()
        pool = [
            c.name for c in get_models_by_role("paraphraser")
            if not (c.name in seen or seen.add(c.name))
        ]
        if not pool:
            raise ValueError("paraphraser='distribution' but no model has role='paraphraser'.")
        return pool
    if paraphraser:
        return [paraphraser]
    return [default_model_for_role("paraphraser")]


def _assign_paraphraser(base_id: str, k: int, pool: list[str], seed: int) -> str:
    """Deterministic round-robin pick (content-hash of base_id + k + seed)."""
    if len(pool) == 1:
        return pool[0]
    h = hashlib.sha256(f"{seed}:{base_id}:{k}".encode()).hexdigest()
    return pool[int(h, 16) % len(pool)]


def _stop_on_an_earlier_folder(ledger: Ledger, out_path: Path) -> None:
    """Raise when the outputs target's folder was written before its units
    were keyed on the response: an ack without ``response_id``, or a file
    whose header lacks one of the response's columns.

    The old acks are read through a ledger keyed on ``iteration``, a field
    both shapes have: the stage's own ledger skips a record without its key.

    Raises:
        ValueError: What was found, and the remedies.
    """
    remedy = ('Paraphrase the outputs over with resume=False (a recipe\'s "resume": '
              'false; params paraphrase.target: "outputs" leaves the paraphrases '
              "of the inputs as they are), or run the stage in a new data_dir.")
    earlier = Ledger(ledger.path, key_fields=("iteration",))
    if any("response_id" not in ack for ack in earlier.records()):
        raise ValueError(
            f"{ledger.path} acks units without a response_id: its paraphrases of "
            "outputs were keyed on the input answered, before an input had "
            f"several responses. {remedy}")
    if out_path.exists():
        try:
            check_columns(out_path, [dict.fromkeys(_RESPONSE_COLUMNS, "")])
        except ValueError as error:
            raise ValueError(
                f"{error} It was written before the paraphrases of outputs were "
                f"keyed on their response. {remedy}") from None


def run_paraphrase_target(
    target: str,
    source: pd.DataFrame,
    out_path: Path,
    *,
    pool: list[str],
    check: ModelClient | None,
    k: int,
    resume: bool = True,
    batch_size: int,
    seed: int,
    prompt_dir: str | Path | None = None,
    verbose: bool = True,
    read_back: bool = True,
    min_words: int | None = None,
) -> pd.DataFrame:
    """Paraphrase each row of ``source`` ``k`` times into ``out_path``.

    ``target`` (``"inputs"`` or ``"outputs"``) says which columns hold a
    row's text and id. A unit of the inputs target is ``(input_id,
    iteration)``, ``input_id`` the input's ``sample_id``; one of the outputs
    target is ``(response_id, iteration)``, ``response_id`` the response's
    ``sample_id``, since an input may have several responses. A paraphrase of
    a response keeps ``input_id``, the input answered, and always carries
    ``response_id``, ``response_mode``, ``output_label`` and
    ``response_label`` (empty when ``source`` lacks the column). Each unit
    goes to a paraphraser of ``pool``, picked by a hash of its id;
    ``check``, when given, gates each paraphrase and judges that it keeps
    its original's meaning. A unit whose text has under ``min_words`` words
    (``None``: :data:`MIN_WORDS`, read at the call; 0 sends every text) gets
    no call: it is acked ``skipped`` with its ``reason`` and the threshold,
    and writes no row. A unit acked ``skipped`` stays skipped on resume
    whatever the current threshold (acks win, with a WARNING when the
    threshold would now send it); ``resume=False`` plans anew. Returns the
    artifact's rows, or with ``read_back=False`` an empty frame without
    reading them.

    Raises:
        ValueError: For the outputs target, a folder written before its units
            were keyed on the response (an ack without ``response_id``, or a
            file without the response's columns), before any call.
    """
    if min_words is None:
        min_words = MIN_WORDS
    outputs = target != "inputs"
    if not outputs:
        text_col = "sample" if "sample" in source.columns else "prompt"
    else:
        text_col = ("output_response" if "output_response" in source.columns
                    else "sample")
    key_field = "response_id" if outputs else "input_id"

    src = source.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Built before resume_state: a check prompt that does not build stops the
    # stage before it takes the lock or clears anything.
    checker = (build_paraphrase_checker(prompt_dir=prompt_dir)
               if check is not None else None)

    # Every attempted unit is acked, a dropped one too, so a resume skips it.
    ledger = Ledger.sidecar(out_path, key_fields=(key_field, "iteration"),
                            casters={key_field: str, "iteration": int})
    manifest = Manifest.sidecar(out_path)
    # A resume cuts off the rows of a save whose acks did not land: those
    # units are paraphrased again, and none ends with two rows.
    completed: set[tuple[str, int]] = resume_state(
        ledger, resume=resume, artifact=out_path, manifest=manifest,
    )
    if outputs:
        try:
            _stop_on_an_earlier_folder(ledger, out_path)
        except ValueError:
            release_lock(ledger)
            raise
    if source.empty:    # a fresh run has cleared what earlier ones left
        if verbose:
            logger.info("[paraphrase:%s] no source rows; skipping.", target)
        release_lock(ledger)
        return pd.DataFrame()
    # Read after the resume, which leaves only whole rows in the artifact.
    prev = read_rows(out_path)
    seen_texts: set[str] = (
        set(prev["sample"].astype(str))
        if not prev.empty and "sample" in prev.columns
        else set()
    )

    # The plan: every (base_id, iteration) unit and the model it goes to. The
    # base id is the input's sample_id, or the response's for the outputs
    # target, whose rows keep the input answered as input_id.
    ids = row_ids(src, text_col, "sample_id")
    input_ids = row_ids(src, text_col, "input_id") if outputs else ids

    def _cells(column):
        """``column`` of the source as text: empty where it has none."""
        if column not in src.columns:
            return repeat("")
        return ("" if pd.isna(cell) else str(cell) for cell in src[column])

    def _units():
        """Yield the run's units in plan order, built anew on every call.

        A unit is ``(base_id, iteration, model, original_text, category,
        entry_type, input_id, labels)``, ``labels`` the source's
        ``response_mode``, ``output_label`` and ``response_label``.
        No list of them is held: a pass over the units costs time, not memory.
        """
        blank = repeat("")      # for a source without the column
        categories = src.get("category", blank)
        entry_types = src.get("entry_type", blank)
        labels = zip(*(_cells(column) for column in _LABEL_COLUMNS))
        for bid, otext, cat, et, iid, label in zip(
                ids, src[text_col], categories, entry_types, input_ids, labels):
            for iteration in range(k):
                yield (bid, iteration, _assign_paraphraser(bid, iteration, pool, seed),
                       str(otext), str(cat), str(et), str(iid), label)

    def _keyed(bid, iteration, iid) -> dict:
        """A unit's key fields, with the input answered for the outputs target."""
        if outputs:
            return {"response_id": bid, "input_id": iid, "iteration": iteration}
        return {"input_id": bid, "iteration": iteration}

    # The units an earlier run acked skipped: acks win on resume.
    earlier = ({ledger.key(ack) for ack in ledger.records()
                if ack.get("status") == "skipped"} if completed else set())

    def _skipped(bid, iteration, otext) -> bool:
        """Whether the unit is skipped: acked so, or pending and short."""
        if (bid, iteration) in completed:
            return (bid, iteration) in earlier
        return bool(skip_reason(otext, min_words=min_words))

    manifest.write(
        {
            **_keyed(bid, iteration, iid), "paraphrase_model": m,
            "category": cat, "entry_type": et,
            "status": "skipped" if _skipped(bid, iteration, otext) else "planned",
        }
        for bid, iteration, m, otext, cat, et, iid, _label in _units()
    )

    written = 0

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out_path, rows)
        written += len(rows)
        return sizes

    # Acked skips of an earlier run whose text this threshold would send:
    # acks win, so they stay unparaphrased. Counted by key, as acked.
    stale = {(bid, iteration) for bid, iteration, _m, otext, *_ in _units()
             if (bid, iteration) in earlier
             and not skip_reason(otext, min_words=min_words)}
    if stale:
        logger.warning(
            "[paraphrase:%s] %d units acked under an earlier min_words stay "
            "unparaphrased on resume; run with resume: false to paraphrase them.",
            target, len(stale))

    # A text too short to paraphrase: its unit is acked at once, with no call
    # and no row.
    skips: dict[tuple[str, int], dict] = {}
    for bid, iteration, m, otext, _cat, _et, iid, _label in _units():
        reason = skip_reason(otext, min_words=min_words)
        if reason and (bid, iteration) not in completed:
            skips[bid, iteration] = {
                **_keyed(bid, iteration, iid), "paraphrase_model": m,
                "status": "skipped", "reason": reason, "min_words": min_words}
    if skips:
        commit([], append=_append, ledger=ledger, records=skips.values())
        completed |= set(skips)

    # The units left for each paraphraser, in the order its first one comes up.
    planned, groups = 0, {}
    for bid, iteration, m, *_ in _units():
        planned += 1
        if (bid, iteration) not in completed:
            groups[m] = groups.get(m, 0) + 1
    if not groups:
        if verbose and skips:
            logger.info("[paraphrase:%s] %d units under %d words skipped; "
                        "nothing to paraphrase.", target, len(skips), min_words)
        elif verbose:
            logger.info("[paraphrase:%s] nothing to do (all %d units done).",
                        target, planned)
        release_lock(ledger)    # no CommitBuffer will
        return read_rows(out_path) if read_back else pd.DataFrame()
    if verbose and skips:
        logger.info("[paraphrase:%s] %d units under %d words skipped.",
                    target, len(skips), min_words)

    # One paraphraser runs at a time. The residency plan decides whether the
    # pool's engines are unloaded between groups: they stay loaded when the
    # whole pool can be resident together and are all dropped otherwise.
    unload_between = False
    if len(groups) > 1:
        from redact import residency

        # One copy each: a copy of one paraphraser on an idle card would only
        # hold what the next one may need.
        plan = residency.plan_residency(list(groups), copies=dict.fromkeys(groups, 1))
        unload_between = plan.sequential
        # replace=False: a run's own plan, when applied, covers the pool.
        residency.apply_plan(plan, replace=False)
        if verbose and unload_between:
            logger.info(
                "[paraphrase:%s] %d paraphrasers do not co-fit; unloading between groups.",
                target, len(groups),
            )

    # Each paraphraser's units go to the drive_units pool, one paraphraser at
    # a time: their generation and check calls overlap, each unit has its own
    # reply cap, and a finished unit is saved without waiting for the others
    # (every 32 units, and on any exit). A unit's driver key is its index in
    # plan order, since the outputs target may plan two units under one key.
    live: dict[int, tuple] = {}     # index -> (unit, provisional id, client)

    def _save(index, value):
        """A finished unit's row and ack. A paraphrase that is empty, equal to
        its original or to a text already written is dropped; which of two
        equal texts is kept follows the order the units finish in."""
        unit, prov_id, unit_client = live.pop(index)
        bid, iteration, m, otext, cat, et, iid, label = unit
        text, acc, reason = value
        pt = str(text).strip()
        if not pt or pt == otext.strip() or pt in seen_texts:
            buf.add([], {**_keyed(bid, iteration, iid),
                         "paraphrase_model": m, "status": "dropped_dedup"})
            return
        seen_texts.add(pt)
        sample_id = content_id(pt)
        if prov_id is not None:
            unit_client.rename_capture(prov_id, f"{bid}/paraphrase/{sample_id}")
        row = {
            "sample_id": sample_id, **_keyed(bid, iteration, iid),
            "sample": pt, "category": cat, "entry_type": et,
            # Always written for the outputs target, so a first run's header
            # holds them whatever its source has.
            **(dict(zip(_LABEL_COLUMNS, label)) if outputs else {}),
            "paraphrase_model": m, "accepted": bool(acc),
            "reasoning": reason, "source": f"paraphrase_{target}",
        }
        buf.add([row], {**_keyed(bid, iteration, iid), "paraphrase_model": m,
                        "status": "accepted" if acc else "rejected"})

    def _too_long(index, exc):
        """A prompt over a model's context acks its unit with no row: sent
        again, it would be too long again. Any other unit error stops the
        stage."""
        if not isinstance(exc, PromptTooLongError):
            raise exc
        unit, _prov_id, _client = live.pop(index)
        bid, iteration, m, *_rest = unit
        logger.warning("[paraphrase:%s] %s %s, iteration %d, gets no paraphrase: "
                       "its prompt is too long: %s", target, key_field, bid,
                       iteration, exc)
        buf.add([], {**_keyed(bid, iteration, unit[6]), "paraphrase_model": m,
                     "status": "prompt_too_long"})

    def _pending(model, client, template):
        """The pending units of ``model``, each its own generator, built as
        the pool takes them."""
        capture = client.compute_config.supports_internals
        for index, unit in enumerate(_units()):
            if unit[2] != model or unit[:2] in completed:
                continue
            # Captured under the unit, renamed to the paraphrase's sample_id.
            prov_id = f"{unit[0]}/paraphrase/attempt_{unit[1]}" if capture else None
            live[index] = (unit, prov_id, client)
            yield index, _unit_gen(client, check, checker, template, unit[3],
                                   provisional_id=prov_id)

    with CommitBuffer(append=_append, ledger=ledger) as buf:
        for group_i, model in enumerate(groups):
            if group_i and unload_between:
                # The previous client's engine is released next; the check
                # model's is kept.
                client = None
                # Engines of other stages, read now so one loaded meanwhile is kept too.
                resident = residency.loaded_engines(exclude=list(groups))
                residency.unload_local(
                    keep=[check.model] if check is not None else None,
                    engines=resident)
            client = ModelClient.create(model)
            drive_units(
                _pending(model, client, paraphrase_prompt(model, prompt_dir)),
                on_done=_save,
                on_error=_too_long,
                pool_size=pool_size(batch_size, client, check),
                cap=batch_size,
                progress=f"paraphrase:{target} ({model})" if verbose else None,
            )

    if verbose:
        logger.info("[paraphrase:%s] +%d rows (artifact now %d) -> %s",
                    target, written, count_rows(out_path), out_path.name)
    return read_rows(out_path) if read_back else pd.DataFrame()
