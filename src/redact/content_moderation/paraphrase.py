"""The paraphrase stage: K checked paraphrases per row, over a pool of paraphrasers.

Any capable model of the ``paraphraser`` role stands in for the trained
defingerprinting model, which lives in another repository.

Each paraphrase is checked in two steps. Five deterministic gates
(:data:`GATES`: length, split lines, script, language, artifacts) reject
what they can without a model call; the rest go to a separate model at
``llm_pipeline.CHECK_TEMPERATURE`` (0.15), with the strict prompt of
:func:`~redact.content_moderation.checker.build_paraphrase_checker`. The
language gate reads GlotLID from ``jailbreak/obfuscation/langid.py``, the
translation check's identifier, imported where it is used.
"""

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from itertools import islice, repeat
from pathlib import Path

import pandas as pd

from ..dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    content_id,
    count_rows,
    read_rows,
    release_lock,
    resume_state,
    row_ids,
)
from ..llm_pipeline import CHECK_TEMPERATURE, batch_check_samples
from ..llms.client import ModelClient
from ..llms.model_config import default_model_for_role, get_models_by_role
from ..llms.prompting import PromptTemplate
from ..llms.router import batch_generate_samples
from .checker import build_paraphrase_checker

logger = logging.getLogger(__name__)

#TODO(later): move this stage onto llm_pipeline.drive_units, like outputs and
# jailbreaks. Waits for a trained paraphraser; a stand-in model runs it today.

# The columns a paraphrase of a response takes from it: the response's id
# (its unit's key) and how the response answers its input.
_RESPONSE_COLUMNS = ("response_id", "response_mode", "output_label", "response_label")
_LABEL_COLUMNS = _RESPONSE_COLUMNS[1:]


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
    (English answered in Chinese). A text with no letters never fires it."""
    ours = _script_counts(original.strip())
    theirs = _script_counts(paraphrase.strip())
    n_ours, n_theirs = sum(ours.values()), sum(theirs.values())
    if not n_ours or not n_theirs:
        return ""
    foreign = sum(n for script, n in theirs.items()
                  if ours.get(script, 0) / n_ours < SCRIPT_RARE)
    if foreign / n_theirs < SCRIPT_SHARE:
        return ""
    return (f"script changed: {ours.most_common(1)[0][0]} -> "
            f"{theirs.most_common(1)[0][0]}")


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


def paraphrase_batch(
    client: ModelClient,
    samples: list[str],
    prompt_dir: str | None = None,
    progress: str | None = None,
    **kwargs,
) -> list[str]:
    """Paraphrase each sample, in order, in one batch of calls.

    ``progress`` labels the batch; ``kwargs`` go to ``batch_generate_samples``.
    """
    if not samples:
        return []
    template = PromptTemplate.load("paraphrase", "generation", prompt_dir=prompt_dir)
    return batch_generate_samples(
        client, [template(sample=s) for s in samples],
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
    ``check``, when given, judges that a paraphrase keeps its original's
    meaning. Returns the artifact's rows, or with ``read_back=False`` an
    empty frame without reading them.

    Raises:
        ValueError: For the outputs target, a folder written before its units
            were keyed on the response (an ack without ``response_id``, or a
            file without the response's columns), before any call.
    """
    outputs = target != "inputs"
    if not outputs:
        text_col = "sample" if "sample" in source.columns else "prompt"
    else:
        text_col = ("output_response" if "output_response" in source.columns
                    else "sample")
    key_field = "response_id" if outputs else "input_id"

    src = source.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    manifest.write(
        {
            **_keyed(bid, iteration, iid), "paraphrase_model": m,
            "category": cat, "entry_type": et, "status": "planned",
        }
        for bid, iteration, m, _o, cat, et, iid, _label in _units()
    )

    # The units left for each paraphraser, in the order its first one comes up.
    planned, groups = 0, {}
    for bid, iteration, m, *_ in _units():
        planned += 1
        if (bid, iteration) not in completed:
            groups[m] = groups.get(m, 0) + 1
    if not groups:
        if verbose:
            logger.info("[paraphrase:%s] nothing to do (all %d units done).",
                        target, planned)
        release_lock(ledger)    # no CommitBuffer will
        return read_rows(out_path) if read_back else pd.DataFrame()

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

    checker = (build_paraphrase_checker(prompt_dir=prompt_dir)
               if check is not None else None)
    written = 0

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out_path, rows)
        written += len(rows)
        return sizes

    # One save per chunk, at the flush below: ``every`` is not reached before it.
    with CommitBuffer(append=_append, ledger=ledger, every=sum(groups.values())) as buf:
        for group_i, (model, n_units) in enumerate(groups.items()):
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
            # Captured under the unit, renamed to the paraphrase's sample_id below.
            capture_paraphrase = client.compute_config.supports_internals
            gunits = (u for u in _units() if u[2] == model and u[:2] not in completed)
            for _ in range(0, n_units, batch_size):
                chunk = list(islice(gunits, batch_size))
                texts = [u[3] for u in chunk]
                provisional_ids = [
                    f"{bid}/paraphrase/attempt_{iteration}"
                    for bid, iteration, *_ in chunk
                ]
                paraphrased = paraphrase_batch(
                    client, texts,
                    prompt_dir=prompt_dir,
                    progress=f"paraphrase:{target} ({model})" if verbose else None,
                    internals_ids=provisional_ids if capture_paraphrase else None,
                )
                # 1. Per unit, in chunk order: the dedup drop, then the gates.
                # A gated text is a row, so it joins seen_texts as any row does.
                # The gates are part of the check: without one, neither runs.
                kept = []   # (unit, paraphrase, gate reason, provisional id)
                for unit, ptext, prov_id in zip(chunk, paraphrased, provisional_ids):
                    pt = str(ptext).strip()
                    if not pt or pt == unit[3] or pt in seen_texts:
                        kept.append((unit, None, "", prov_id))
                        continue
                    seen_texts.add(pt)
                    reason = gate_reason(unit[3], pt) if check is not None else ""
                    kept.append((unit, pt, reason, prov_id))
                # 2. One check call for the kept, ungated units.
                to_check = [(unit[3], pt) for unit, pt, reason, _ in kept
                            if pt is not None and not reason]
                verdicts = iter(batch_check_samples(
                    check, [pt for _, pt in to_check], checker,
                    originals=[otext for otext, _ in to_check],
                    batch_size=batch_size,
                    progress=f"paraphrase-check:{target}" if verbose else None,
                    temperature=CHECK_TEMPERATURE,
                ) if check is not None and to_check else ())

                # 3. The rows and acks, in chunk order.
                attempted = []  # (rows, ack) of each unit of the chunk
                for unit, pt, gated, prov_id in kept:
                    bid, iteration, m, otext, cat, et, iid, label = unit
                    if pt is None:
                        attempted.append(([], {
                            **_keyed(bid, iteration, iid),
                            "paraphrase_model": m, "status": "dropped_dedup"}))
                        continue  # dedup: drop no-op / duplicate paraphrases
                    if gated:
                        acc, reason = False, gated
                    elif check is not None:
                        acc, reason = next(verdicts)
                    else:
                        acc, reason = True, ""
                    sample_id = content_id(pt)
                    if capture_paraphrase:
                        client.rename_capture(prov_id, f"{bid}/paraphrase/{sample_id}")
                    row = {
                        "sample_id": sample_id, **_keyed(bid, iteration, iid),
                        "sample": pt, "category": cat, "entry_type": et,
                        # Always written for the outputs target, so a first
                        # run's header holds them whatever its source has.
                        **(dict(zip(_LABEL_COLUMNS, label)) if outputs else {}),
                        "paraphrase_model": m, "accepted": bool(acc),
                        "reasoning": reason, "source": f"paraphrase_{target}",
                    }
                    attempted.append(([row], {
                        **_keyed(bid, iteration, iid), "paraphrase_model": m,
                        "status": "accepted" if acc else "rejected"}))
                # Queued once the whole chunk is built: a chunk that fails part-way
                # saves none of its units. The flush appends the rows, then acks
                # every attempted unit, so a dropped one is not attempted again.
                for unit_rows, ack in attempted:
                    buf.add(unit_rows, ack)
                buf.flush()

    if verbose:
        logger.info("[paraphrase:%s] +%d rows (artifact now %d) -> %s",
                    target, written, count_rows(out_path), out_path.name)
    return read_rows(out_path) if read_back else pd.DataFrame()
