"""The jailbreak stage's core: plan, drive the units on the pool, save.

Called once per kind of the augmentation section (the adversarial kind to
``jailbreaks.csv``, the natural kind to ``variations.csv``).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pandas as pd

from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    read_rows,
    resume_state,
)
from redact.llm_pipeline import drive_units, pool_size
from redact.llms.client import ModelClient

from . import pool as pool_module
from .manipulation import get_or_generate_benign_data
from .plan import coverage_path, plan_run
from .technique import load_spec
from .units import error_row_of, is_pending, pending_units, row_of

logger = logging.getLogger(__name__)


def run_augmentation(
    inputs: pd.DataFrame,
    pool: list[Callable],
    *,
    client: ModelClient,
    translation_model: str | None = None,
    out_path: Path,
    manifest_path: Path | None = None,
    benign_path: Path | None = None,
    benign_taxonomy: dict | str = "benign",
    taxonomy_dir: str | Path | None = None,
    prompt_dir=None,
    seed: int = 42,
    rounds: list[dict] | None = None,
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
    auto_generate_benign: bool = True,
    batch_size: int = 256,
    resume: bool = True,
    verbose: bool = True,
    read_back: bool = True,
    label: str = "jailbreak",
    accept_noop: bool = True,
) -> pd.DataFrame:
    """Augment the inputs with ``pool``'s techniques, appended to ``out_path``.

    The run is planned whole into a manifest beside ``out_path`` — one
    combination per input and round, drawn from ``pool`` and frozen — then
    its pending units run through :func:`drive_units`, up to ``batch_size`` per
    call, their rows appended every 32 units. The combination is read back
    through every registered technique, so a plan made with another pool still
    replays, and with ``auto_generate_benign`` the benign data is loaded when
    a pending unit's techniques need it, its pairs generated first for each
    subcategory of ``benign_taxonomy`` (a file name under ``taxonomy_dir``, or
    a loaded taxonomy) that ``benign_path`` lacks. ``rounds`` holds each round's
    sampler settings, as :func:`redact.jailbreak.augmentation.rounds_of`
    gives them (one round of the sampler's defaults when None);
    ``family_weights`` and ``layer_rates``
    go to the dealing of :func:`plan_run`. ``label`` is the progress label and
    the internals-capture root. With ``accept_noop`` False, a row whose text
    is its input's is not accepted.

    Resume: completion is a sidecar ``*.state.jsonl`` ledger keyed
    ``(input_id, iteration)``; ``resume=True`` reuses the plan and skips its
    finished units, ``resume=False`` clears the output, ledger and plan first.

    Returns:
        The rows of ``out_path``, or an empty frame when ``read_back`` is False.
    """
    out_path = Path(out_path)
    man_path = Path(manifest_path) if manifest_path else Manifest.sidecar(out_path).path
    rounds = [{}] if rounds is None else list(rounds)
    if not rounds:
        raise ValueError("rounds must be a non-empty list of dicts.")

    # The registry is every registered technique, not the (filtered) pool, so a
    # plan made with another pool still reconstructs.
    registry = pool_module.registry(pool_module.all_techniques())
    capture = client.compute_config.supports_internals

    spec_version = load_spec().get("version", "")

    if verbose:
        logger.info("%s (plan + batched execute)", label)
        per_layer = Counter(getattr(fn, "layer", None) for fn in pool)
        logger.info(
            "Pool: %d techniques (%s) | Samples: %d | Rounds: %d | Model: %s | "
            "Batch: %d", len(pool),
            ", ".join(f"{layer} {n}" for layer, n in per_layer.items()),
            len(inputs), len(rounds), client.model, batch_size)
        if layer_rates or family_weights:
            logger.info("Layer rates: %s | Family weights: %s",
                        layer_rates or "the spec's", family_weights or "equal")

    ledger = Ledger.sidecar(
        out_path, key_fields=("input_id", "iteration"),
        casters={"input_id": str, "iteration": int},
    )

    # Before the plan is written: the resume takes the stage's lock. A fresh
    # (non-resume) run clears prior output, ledger and plan, so a run stopped
    # while it plans does not leave the old plan to be reused; a resume keeps
    # and skips what the ledger records, and cuts off an unfinished save.
    manifest = Manifest(man_path)
    completed = resume_state(
        ledger, resume=resume, artifact=out_path, manifest=manifest)

    if resume and man_path.exists():
        if verbose:
            logger.info("Reusing existing manifest: %s", man_path)
    else:
        coverage_path(man_path).unlink(missing_ok=True)
        plan_run(
            inputs, pool, manifest_path=man_path, seed=seed,
            iterations=len(rounds), settings_per_iteration=rounds,
            family_weights=family_weights,
            layer_rates=layer_rates, verbose=verbose,
        )

    # Where each input is in the frame, by its id; of two inputs with one id
    # the later one. A unit's prompt and its row's input columns are read there
    # when needed, so no copy of every input is held.
    position = dict(zip(inputs["sample_id"], range(len(inputs))))
    prompts = inputs["prompt"]

    # One pass over the plan, for the numbers in the log lines and for whether
    # a pending unit's techniques need benign data: read off the plan, not the
    # pool, since a plan made with another pool still replays.
    planned = total = written = 0
    needs_benign = False
    if verbose or auto_generate_benign:
        for plan_row in manifest:
            planned += 1
            if is_pending(plan_row, position, completed):
                total += 1
                needs_benign = needs_benign or any(
                    getattr(registry.get(name), "requires_benign", False)
                    for name in plan_row["combination"])
    benign_data = None
    if auto_generate_benign and needs_benign:
        benign_data = get_or_generate_benign_data(
            client=client, cache_path=benign_path, verbose=verbose,
            taxonomy=benign_taxonomy, taxonomy_dir=taxonomy_dir,
            prompt_dir=prompt_dir, batch_size=batch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose and total:
        logger.info("Executing %d units...", total)

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out_path, rows)
        written += len(rows)
        if verbose:
            n_acc = sum(1 for x in rows if x.get("accepted"))
            logger.info("[%d/%d] %d accepted -> appended to %s",
                        written, total, n_acc, out_path.name)
        return sizes

    live: dict[int, dict] = {}

    def _row(unit: dict, make: Callable) -> None:
        input_row = inputs.iloc[position[unit["id"]]].to_dict()
        buf.add([make(unit, input_row)], {"input_id": str(unit["id"]),
                                          "iteration": int(unit["iteration"])})

    def _on_done(i, value):
        _row(live.pop(i),
             lambda u, r: row_of(u, value, r, spec_version, gen_model=client.model,
                                 accept_noop=accept_noop))

    def _on_error(i, exc):
        _row(live.pop(i),
             lambda u, r: error_row_of(u, exc, r, spec_version, gen_model=client.model))

    def _units():
        # The plan is read one row at a time, as the driver takes units; closed
        # with this generator, since Windows does not replace an open file.
        with closing(iter(manifest)) as plan:
            yield from pending_units(
                plan, prompts, position, completed, registry, live,
                gen_model=client.model, translate_model=translation_model,
                benign_data=benign_data, prompt_dir=prompt_dir,
                capture=capture, label=label,
            )

    # A unit whose chain raises gets an error row and is acked. The buffer
    # appends, then acks (see dataset/save.py), every 32 units and on any exit.
    with (
        closing(_units()) as units,
        CommitBuffer(append=_append, ledger=ledger) as buf,
    ):
        drive_units(
            units, on_done=_on_done, on_error=_on_error,
            pool_size=pool_size(batch_size, client), cap=batch_size,
            progress=label if verbose else None,
        )

    rows = read_rows(out_path) if read_back else pd.DataFrame()

    if verbose:
        logger.info("%s summary", label)
        logger.info("Planned units: %d | Newly written: %d", planned, written)
        if read_back:
            logger.info("Total rows in %s: %d", out_path.name, len(rows))
        if not rows.empty and "accepted" in rows.columns:
            n_accepted = int(rows["accepted"].sum())
            n_noop = int(rows["is_noop"].sum()) if "is_noop" in rows.columns else 0
            logger.info("Accepted: %d | Rejected: %d",
                        n_accepted, len(rows) - n_accepted)
            logger.info("No-ops: %d", n_noop)
        logger.info("Manifest: %s | Saved to: %s", man_path, out_path)

    return rows
