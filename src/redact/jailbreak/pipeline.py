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
    release_lock,
    resume_state,
    row_ids,
)
from redact.dataset.planfile import changed_plan, plan_path, read_plan, variants_path
from redact.llm_pipeline import drive_units, pool_size
from redact.llms.client import ModelClient

from . import pool as pool_module
from .manipulation import get_or_generate_benign_data
from .plan import coverage_path, plan_inputs, plan_run, rules_id
from .rules import compatible
from .technique import FIRST_VERSION, load_spec, version_key
from .units import error_row_of, is_pending, pending_units, row_of
from .variants import write_variants

logger = logging.getLogger(__name__)


def run_augmentation(
    inputs: pd.DataFrame,
    pool: list[Callable],
    *,
    client: ModelClient,
    check: ModelClient | None = None,
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
    replays. When the units are done, ``<stem>.variants.csv`` beside the
    manifest counts the template variants the rows of ``out_path`` drew
    (:func:`~redact.jailbreak.variants.write_variants`). The benign data at
    ``benign_path`` is loaded when a pending unit's techniques need it: with
    ``auto_generate_benign`` its pairs are generated first for each
    subcategory of ``benign_taxonomy`` (a file name under ``taxonomy_dir``,
    or a loaded taxonomy) that the file lacks; without it a missing file
    raises before any unit runs. ``rounds`` holds each round's
    sampler settings, as :func:`redact.jailbreak.augmentation.rounds_of`
    gives them (one round of the sampler's defaults when None);
    ``family_weights`` and ``layer_rates``
    go to the dealing of :func:`plan_run`. ``check`` checks what a
    technique's model writes (the typo check), ``client`` when None;
    translation keeps its own checkers. ``label`` is the progress label, the
    internals-capture root, and part of the techniques' random seed and of an
    error row's id. With ``accept_noop`` False, a row whose text
    is its input's is not accepted.

    Each input's id is its ``sample_id`` when it has one, else the content id
    of its prompt. Of rows that share an id, the last is kept, with a
    WARNING (:func:`_one_row_per_id`); rows that share one under different
    entry types raise a ``ValueError`` before anything is written.

    Resume: completion is a sidecar ``*.state.jsonl`` ledger keyed
    ``(input_id, iteration)``; ``resume=True`` reuses the plan and skips its
    finished units, ``resume=False`` clears the output, ledger and plan (its
    plan file, coverage and variants tables too) first.
    A reused plan runs as it was made: an input it does not hold, accepted
    after it was made, gets no unit, and a WARNING counts those inputs. A
    reused plan made with another ``seed``, ``rounds``, pool, ``family_weights``
    or ``layer_rates`` (its plan file says) raises a ``ValueError``; one with
    no plan file is run unchecked, with a WARNING. A pool that differs only
    by what the library added or retired after the plan's spec version
    (:func:`_library_change`) is no change of this run's: the plan is kept,
    with a WARNING that names those techniques, so a kept plan resumes in a
    library that grew. The same one WARNING is given when the plan's spec
    version or the rules it was drawn by (its plan file's ``rules``,
    :func:`~redact.jailbreak.plan.rules_id`; unknown in a plan file written
    before it recorded them) are not today's, and counts the pending units
    whose combination today's rules refuse; they run as planned. A row
    records the spec version its plan was drawn at
    (``combination_spec_version``), not the library's; a plan without a
    plan file records none, and its rows the library's.

    Returns:
        The rows of ``out_path``, or an empty frame when ``read_back`` is False.
    """
    inputs = _one_row_per_id(inputs.assign(sample_id=row_ids(inputs, "prompt")), label)
    out_path = Path(out_path)
    man_path = Path(manifest_path) if manifest_path else Manifest.sidecar(out_path).path
    rounds = [{}] if rounds is None else list(rounds)
    if not rounds:
        raise ValueError("rounds must be a non-empty list of dicts.")

    # The registry is every registered technique, not the (filtered) pool, so a
    # plan made with another pool still reconstructs.
    registry = pool_module.registry(pool_module.all_techniques())
    capture = client.compute_config.supports_internals
    check = check or client

    spec_version = load_spec().get("version", "")

    if verbose:
        logger.info("%s (plan + batched execute)", label)
        per_layer = Counter(getattr(fn, "layer", None) for fn in pool)
        logger.info(
            "Pool: %d techniques (%s) | Samples: %d | Rounds: %d | Model: %s | "
            "Check: %s | Batch: %d", len(pool),
            ", ".join(f"{layer} {n}" for layer, n in per_layer.items()),
            len(inputs), len(rounds), client.model, check.model, batch_size)
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

    reused = resume and man_path.exists()
    # The spec version the plan was drawn at, which its rows record: this
    # library's for a fresh plan, the plan file's for a kept one ("1.0" in a
    # plan file written before it recorded one), and this library's again
    # for a kept plan without a plan file, which records nothing.
    plan_version = spec_version
    # What the library changed since a kept plan was made, said in one
    # WARNING once the pass over the plan has counted the pending units
    # whose combination today's rules refuse.
    since_plan = None
    if reused:
        if verbose:
            logger.info("Reusing existing manifest: %s", man_path)
        # changed_plan compares every key of the record it is given, so it
        # gets the record less the pool, the spec version and the rules,
        # which are compared below: a pool that differs only by what the
        # library added or retired after the plan's spec_version is no
        # change of this run's, spec_version only dates the pool (a plan
        # file written before it was recorded has none), and the rules are
        # the library's, not the run's.
        current = plan_inputs(
            pool, seed=seed, rounds=rounds, family_weights=family_weights,
            layer_rates=layer_rates)
        changed = changed_plan(man_path, {
            key: value for key, value in current.items()
            if key not in ("pool", "spec_version", "rules")})
        if changed is None:
            logger.warning(
                "%s: the plan %s has no %s beside it, so whether this run's "
                "section and seed are the ones it was made with is not checked; "
                "it runs as it was made. Run with resume=False to plan again "
                "(the stage starts over).", label, man_path.name,
                plan_path(man_path).name)
        else:
            made_with = read_plan(man_path)
            plan_version = made_with.get("spec_version", FIRST_VERSION)
            library = _library_change(made_with, pool, registry)
            if library is None:
                changed.append("pool")
            if changed:
                release_lock(ledger)
                raise ValueError(
                    f"{label}: this run's {changed} differ from those the plan "
                    f"{man_path.name} was made with (see "
                    f"{plan_path(man_path).name}), and a resume keeps the plan "
                    "as it was made. Run with resume=False to plan again (the "
                    "stage starts over), or with the section and seed the plan "
                    "was made with.")
            added, retired = library
            rules = _rules_change(made_with, registry)
            if added or retired or rules or plan_version != spec_version:
                since_plan = {"added": added, "retired": retired, "rules": rules,
                              "refused": 0}
    else:
        coverage_path(man_path).unlink(missing_ok=True)
        plan_path(man_path).unlink(missing_ok=True)
        variants_path(man_path).unlink(missing_ok=True)
        plan_run(
            inputs, pool, manifest_path=man_path, seed=seed,
            iterations=len(rounds), settings_per_iteration=rounds,
            family_weights=family_weights,
            layer_rates=layer_rates, verbose=verbose,
        )

    # Where each input is in the frame, by its id (one row per id, by now).
    # A unit's prompt and its row's input columns are read there when
    # needed, so no copy of every input is held.
    position = dict(zip(inputs["sample_id"], range(len(inputs))))
    prompts = inputs["prompt"]

    # One pass over the plan, for the numbers in the log lines, for whether
    # a pending unit's techniques need benign data (read off the plan, not the
    # pool, since a plan made with another pool still replays), for the
    # inputs a reused plan does not hold, and for the pending units whose
    # combination today's rules refuse.
    planned = total = written = 0
    needs_benign = False
    unplanned = set(position) if reused else set()
    pool_needs_benign = any(getattr(fn, "requires_benign", False) for fn in pool)
    if verbose or pool_needs_benign or reused:
        for plan_row in manifest:
            planned += 1
            unplanned.discard(str(plan_row["sample_id"]))
            if is_pending(plan_row, position, completed):
                total += 1
                needs_benign = needs_benign or any(
                    getattr(registry.get(name), "requires_benign", False)
                    for name in plan_row["combination"])
                if since_plan is not None:
                    since_plan["refused"] += _refused(plan_row["combination"], registry)
    if since_plan is not None:
        _warn_since_plan(label, man_path.name, plan_version, spec_version,
                         pending=total, **since_plan)
    if unplanned:
        logger.warning(
            "%s: %d of the %d inputs are not in the plan %s, which a resume "
            "keeps as it was made; they were accepted after it, and get no "
            "unit. Run with resume=False to plan every input again (the stage "
            "starts over).", label, len(unplanned), len(position), man_path.name)
    benign_data = None
    if needs_benign:
        benign_data = get_or_generate_benign_data(
            client=client if auto_generate_benign else None,
            cache_path=benign_path, verbose=verbose,
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
             lambda u, r: row_of(u, value, r, plan_version, gen_model=client.model,
                                 accept_noop=accept_noop))

    def _on_error(i, exc):
        _row(live.pop(i),
             lambda u, r: error_row_of(u, exc, r, plan_version, gen_model=client.model,
                                       label=label))

    def _units():
        # The plan is read one row at a time, as the driver takes units; closed
        # with this generator, since Windows does not replace an open file.
        with closing(iter(manifest)) as plan:
            yield from pending_units(
                plan, prompts, position, completed, registry, live,
                gen_model=client.model, translate_model=translation_model,
                check_model=check.model,
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
            pool_size=pool_size(batch_size, client, check), cap=batch_size,
            progress=label if verbose else None,
        )

    # Read from the file, rows of earlier runs included: the natural kind
    # runs with read_back False.
    variants = write_variants(out_path, man_path, pool, registry)
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
        logger.info("Manifest: %s | Saved to: %s | Variants: %s",
                    man_path, out_path, variants)

    return rows


def _one_row_per_id(inputs: pd.DataFrame, label: str) -> pd.DataFrame:
    """``inputs`` with one row per ``sample_id``. Rows that share one (one
    text without an id of its own, or an id given twice) would be planned
    as two units under one ledger key, the row of each written with the
    last one's columns; the last is kept, with a WARNING.

    Raises:
        ValueError: Rows that share an id have different entry types, so
            one of the two labels would be wrong.
    """
    repeated = inputs["sample_id"].duplicated(keep=False)
    if not repeated.any():
        return inputs
    if "entry_type" in inputs.columns:
        types = inputs.loc[repeated].groupby("sample_id")["entry_type"].nunique(
            dropna=False)
        mixed = sorted(types[types > 1].index)
        if mixed:
            raise ValueError(
                f"{label}: {len(mixed)} input id(s) are each held by rows of "
                f"different entry types ({mixed[:3]}), the same text or the same "
                "id under two labels; keep one row of each in the inputs, or "
                "give the rows ids of their own.")
    kept = inputs.drop_duplicates("sample_id", keep="last")
    logger.warning(
        "%s: %d input rows share their id with another (the same text, or an "
        "id given twice); the plan keeps the last row of each id and leaves "
        "out %d.", label, int(repeated.sum()), len(inputs) - len(kept))
    return kept


def _library_change(
    made_with: dict, pool: list[Callable], registry: dict[str, Callable],
) -> tuple[list[str], list[str]] | None:
    """What the library changed in ``pool`` since the plan whose plan file
    holds ``made_with``: ``(added, retired)``, the names it added after the
    plan's ``spec_version`` (a later ``since``) and those it retired after
    it (a later ``deprecated``), each sorted; both empty when the pool is
    the plan's. ``None`` when the pool differs otherwise: a technique this
    run's section adds or leaves out, or one no longer registered.

    A plan file without ``spec_version``, written before it was recorded,
    was made at :data:`FIRST_VERSION`. Known limit: a section that newly
    names a family all of whose techniques came after the plan reads as the
    library's growth too, and warns instead of raising.
    """
    made_at = version_key(made_with.get("spec_version", FIRST_VERSION))

    def after(version) -> bool:
        return version is not None and version_key(version) > made_at

    planned = set(made_with.get("pool", ()))
    current = {fn.__name__: fn for fn in pool}
    added = sorted(set(current) - planned)
    retired = sorted(planned - set(current))
    if (all(after(getattr(current[name], "since", FIRST_VERSION)) for name in added)
            and all(after(getattr(registry.get(name), "deprecated", None))
                    for name in retired)):
        return added, retired
    return None


def _rules_change(made_with: dict, registry: dict[str, Callable]) -> str | None:
    """How the rules the plan whose plan file holds ``made_with`` was drawn
    by compare with today's, for the techniques of its pool
    (:func:`~redact.jailbreak.plan.rules_id`): ``None`` when they are the
    same, ``"changed"`` when they differ, and ``"unknown"`` when the plan
    file, written before it recorded them, has no ``rules``."""
    if "rules" not in made_with:
        return "unknown"
    planned = [registry[name] for name in made_with.get("pool", ()) if name in registry]
    return None if rules_id(planned) == made_with["rules"] else "changed"


def _refused(names: list[str], registry: dict[str, Callable]) -> bool:
    """Whether today's rules refuse the combination ``names``: one of its
    techniques may not join those before it (:func:`compatible`; the rules
    bind both ways, so the order does not matter). A name no technique has
    is left to the replay, which raises on it."""
    if any(name not in registry for name in names):
        return False
    fns = [registry[name] for name in names]
    return any(not compatible(fns[:i], [fn]) for i, fn in enumerate(fns))


def _warn_since_plan(
    label: str, plan_name: str, made_at: str, now: str, *,
    added: list[str], retired: list[str], rules: str | None, refused: int,
    pending: int,
) -> None:
    """The one WARNING of a kept plan the library changed since: the spec
    versions it was made at and is resumed at, what the library added and
    retired, whether its rules changed or are not recorded (``rules``, as
    :func:`_rules_change` gives it), and the ``refused`` of its ``pending``
    units whose combination today's rules refuse."""
    logger.warning(
        "%s: since the plan %s was made (spec version %s, now %s), the library "
        "added %s and retired %s%s; %d of its %d pending units hold a "
        "combination today's rules refuse. A resume keeps the plan as it was "
        "made: it does not draw the added techniques, still runs the retired "
        "ones and the refused combinations it names, and its rows record spec "
        "version %s. Run with resume=False to plan again (the stage starts "
        "over).", label, plan_name, made_at, now, added, retired,
        ", and changed the rules a combination is drawn by" if rules == "changed"
        else ", and the plan file does not record the rules it was drawn by"
        if rules == "unknown" else "",
        refused, pending, made_at)
