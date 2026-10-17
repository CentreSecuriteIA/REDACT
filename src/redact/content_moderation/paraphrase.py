"""The paraphrase stage: K checked paraphrases per row, over a pool of paraphrasers.

Any capable model of the ``paraphraser`` role stands in for the trained
defingerprinting model, which lives in another repository. A separate model
checks that each paraphrase keeps the original's meaning
(:func:`~redact.content_moderation.checker.build_paraphrase_checker`).
"""

import hashlib
import logging
from itertools import islice, repeat
from pathlib import Path

import pandas as pd

from ..dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    content_id,
    count_rows,
    read_rows,
    release_lock,
    resume_state,
    row_ids,
)
from ..llm_pipeline import batch_check_samples
from ..llms.client import ModelClient
from ..llms.model_config import default_model_for_role, get_models_by_role
from ..llms.prompting import PromptTemplate
from ..llms.router import batch_generate_samples
from .checker import build_paraphrase_checker

logger = logging.getLogger(__name__)

#TODO(later): move this stage onto llm_pipeline.drive_units, like outputs and
# jailbreaks. Waits for a trained paraphraser; a stand-in model runs it today.


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
    row's text and id. Each unit goes to a paraphraser of ``pool``;
    ``check``, when given, judges that a paraphrase keeps its original's
    meaning. Returns the artifact's rows, or with ``read_back=False`` an
    empty frame without reading them.
    """
    if source.empty:
        if verbose:
            logger.info("[paraphrase:%s] no source rows; skipping.", target)
        return pd.DataFrame()
    if target == "inputs":
        text_col = "sample" if "sample" in source.columns else "prompt"
        id_col = "sample_id"
    else:
        text_col = ("output_response" if "output_response" in source.columns
                    else "sample")
        id_col = "input_id"

    src = source.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Every attempted unit is acked, a dropped one too, so a resume skips it.
    ledger = Ledger.sidecar(out_path, key_fields=("input_id", "iteration"),
                            casters={"input_id": str, "iteration": int})
    manifest = Manifest.sidecar(out_path)
    # A resume cuts off the rows of a save whose acks did not land: those
    # units are paraphrased again, and none ends with two rows.
    completed: set[tuple[str, int]] = resume_state(
        ledger, resume=resume, artifact=out_path, manifest=manifest,
    )
    # Read after the resume, which leaves only whole rows in the artifact.
    prev = read_rows(out_path)
    seen_texts: set[str] = (
        set(prev["sample"].astype(str))
        if not prev.empty and "sample" in prev.columns
        else set()
    )

    # The plan: every (base_id, iteration) unit and the model it goes to.
    ids = row_ids(src, text_col, id_col)

    def _units():
        """Yield the run's units in plan order, built anew on every call.

        A unit is ``(base_id, iteration, model, original_text, category,
        entry_type)``.
        No list of them is held: a pass over the units costs time, not memory.
        """
        blank = repeat("")      # for a source without the column
        categories = src.get("category", blank)
        entry_types = src.get("entry_type", blank)
        for bid, otext, cat, et in zip(ids, src[text_col], categories, entry_types):
            for iteration in range(k):
                yield (bid, iteration, _assign_paraphraser(bid, iteration, pool, seed),
                       str(otext), str(cat), str(et))

    manifest.write(
        {
            "input_id": bid, "iteration": iteration, "paraphrase_model": m,
            "category": cat, "entry_type": et, "status": "planned",
        }
        for bid, iteration, m, _o, cat, et in _units()
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
                if check is not None:
                    originals = [u[3] for u in chunk]
                    checks = batch_check_samples(
                        check, paraphrased,
                        build_paraphrase_checker(prompt_dir=prompt_dir),
                        originals=originals,
                        batch_size=batch_size,
                        progress=f"paraphrase-check:{target}" if verbose else None,
                    )
                else:
                    checks = [(True, "")] * len(chunk)

                attempted = []  # (rows, ack) of each unit of the chunk
                for unit, ptext, (acc, reason), prov_id in zip(
                    chunk, paraphrased, checks, provisional_ids
                ):
                    bid, iteration, m, otext, cat, et = unit
                    pt = str(ptext).strip()
                    if not pt or pt == otext or pt in seen_texts:
                        attempted.append(([], {
                            "input_id": bid, "iteration": iteration,
                            "paraphrase_model": m, "status": "dropped_dedup"}))
                        continue  # dedup: drop no-op / duplicate paraphrases
                    seen_texts.add(pt)
                    sample_id = content_id(pt)
                    if capture_paraphrase:
                        client.rename_capture(prov_id, f"{bid}/paraphrase/{sample_id}")
                    row = {
                        "sample_id": sample_id, "input_id": bid,
                        "iteration": iteration, "sample": pt,
                        "category": cat, "entry_type": et,
                        "paraphrase_model": m, "accepted": bool(acc),
                        "reasoning": reason, "source": f"paraphrase_{target}",
                    }
                    attempted.append(([row], {
                        "input_id": bid, "iteration": iteration, "paraphrase_model": m,
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
