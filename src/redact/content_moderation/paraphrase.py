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

from .. import paths
from ..dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    content_id,
    count_rows,
    merge_inputs,
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
    tgt, source, data_dir, pool, check, check_model, K, resume, batch_size,
    seed, prompt_dir, out_path_override, verbose, read_back=True,
) -> pd.DataFrame:
    """Paraphrase one target (``"inputs"`` or ``"outputs"``) → its artifact CSV.

    Returns the artifact's rows, or with ``read_back=False`` an empty frame
    without reading them.
    """
    if tgt == "inputs":
        src = source if source is not None else merge_inputs(data_dir)
        text_col = "sample" if "sample" in src.columns else "prompt"
        id_col = "sample_id"
        out_path = Path(out_path_override) if out_path_override else paths.paraphrases_inputs_csv(data_dir)
    else:  # outputs
        src = (source if source is not None
               else read_rows(paths.output_responses_csv(data_dir)))
        if not src.empty and "accepted" in src.columns:
            src = src[src["accepted"] == True]  # noqa: E712 — accepted base outputs only
        text_col = "output_response" if "output_response" in getattr(src, "columns", []) else "sample"
        id_col = "input_id"
        out_path = Path(out_path_override) if out_path_override else paths.paraphrases_outputs_csv(data_dir)

    if src is None or src.empty:
        if verbose:
            logger.info("[paraphrase:%s] no source rows; skipping.", tgt)
        return pd.DataFrame()

    src = src.reset_index(drop=True)
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

    # The plan: every (base_id, k) unit and the model it goes to.
    ids = row_ids(src, text_col, id_col)

    def _units():
        """Yield the run's units in plan order, built anew on every call.

        A unit is ``(base_id, k, model, original_text, category, entry_type)``.
        No list of them is held: a pass over the units costs time, not memory.
        """
        blank = repeat("")      # for a source without the column
        categories = src.get("category", blank)
        entry_types = src.get("entry_type", blank)
        for bid, otext, cat, et in zip(ids, src[text_col], categories, entry_types):
            for k in range(K):
                yield (bid, k, _assign_paraphraser(bid, k, pool, seed),
                       str(otext), str(cat), str(et))

    manifest.write(
        {
            "input_id": bid, "iteration": k, "paraphrase_model": m,
            "category": cat, "entry_type": et, "status": "planned",
        }
        for bid, k, m, _o, cat, et in _units()
    )

    # The units left for each paraphraser, in the order its first one comes up.
    planned, groups = 0, {}
    for bid, k, m, *_ in _units():
        planned += 1
        if (bid, k) not in completed:
            groups[m] = groups.get(m, 0) + 1
    if not groups:
        if verbose:
            logger.info("[paraphrase:%s] nothing to do (all %d units done).", tgt, planned)
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
                tgt, len(groups),
            )

    check_client = ModelClient.create(check_model) if check else None
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
                residency.unload_local(keep=[check_model] if check else None,
                                       engines=resident)
            client = ModelClient.create(model)
            # Captured under bid/k, renamed to the paraphrase's sample_id below.
            capture_paraphrase = client.compute_config.supports_internals
            gunits = (u for u in _units() if u[2] == model and u[:2] not in completed)
            for _ in range(0, n_units, batch_size):
                chunk = list(islice(gunits, batch_size))
                texts = [u[3] for u in chunk]
                provisional_ids = [
                    f"{bid}/paraphrase/attempt_{k}" for bid, k, *_ in chunk
                ]
                paraphrased = paraphrase_batch(
                    client, texts,
                    prompt_dir=prompt_dir,
                    progress=f"paraphrase:{tgt} ({model})" if verbose else None,
                    internals_ids=provisional_ids if capture_paraphrase else None,
                )
                if check:
                    originals = [u[3] for u in chunk]
                    checks = batch_check_samples(
                        check_client, paraphrased,
                        build_paraphrase_checker(prompt_dir=prompt_dir),
                        originals=originals,
                        batch_size=batch_size,
                        progress=f"paraphrase-check:{tgt}" if verbose else None,
                    )
                else:
                    checks = [(True, "")] * len(chunk)

                attempted = []  # (rows, ack) of each unit of the chunk
                for (bid, k, m, otext, cat, et), ptext, (acc, reason), prov_id in zip(
                    chunk, paraphrased, checks, provisional_ids
                ):
                    pt = str(ptext).strip()
                    if not pt or pt == otext or pt in seen_texts:
                        attempted.append(([], {
                            "input_id": bid, "iteration": k,
                            "paraphrase_model": m, "status": "dropped_dedup"}))
                        continue  # dedup: drop no-op / duplicate paraphrases
                    seen_texts.add(pt)
                    sample_id = content_id(pt)
                    if capture_paraphrase:
                        client.rename_capture(prov_id, f"{bid}/paraphrase/{sample_id}")
                    row = {
                        "sample_id": sample_id, "input_id": bid, "iteration": k,
                        "sample": pt, "category": cat, "entry_type": et,
                        "paraphrase_model": m, "accepted": bool(acc),
                        "reasoning": reason, "source": f"paraphrase_{tgt}",
                    }
                    attempted.append(([row], {
                        "input_id": bid, "iteration": k, "paraphrase_model": m,
                        "status": "accepted" if acc else "rejected"}))
                # Queued once the whole chunk is built: a chunk that fails part-way
                # saves none of its units. The flush appends the rows, then acks
                # every attempted unit, so a dropped one is not attempted again.
                for unit_rows, ack in attempted:
                    buf.add(unit_rows, ack)
                buf.flush()

    if verbose:
        logger.info("[paraphrase:%s] +%d rows (artifact now %d) -> %s",
                    tgt, written, count_rows(out_path), out_path.name)
    return read_rows(out_path) if read_back else pd.DataFrame()
