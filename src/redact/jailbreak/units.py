"""A planned unit as the driver runs it: its generator, and its row when it ends."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from redact.dataset import content_id

from .chain import run_chain
from .plan import combination_of
from .technique import is_discarded


def is_pending(plan_row: dict, position: dict, completed: set) -> bool:
    """Whether a planned unit is of these inputs and not yet finished."""
    sid = str(plan_row["sample_id"])
    return sid in position and (sid, int(plan_row["iteration"])) not in completed


def unit_gen(
    unit: dict,
    *,
    gen_model: str,
    translate_model: str | None = None,
    benign_data: dict | None = None,
    prompt_dir=None,
    capture: bool = False,
    label: str = "jailbreak",
) -> Iterator:
    """The chain generator of one unit.

    Each technique draws its random choices from an ``rng`` seeded by the
    plan's seed, ``label``, the unit's id and iteration, and the technique's
    name, so a replayed plan gives the same text and the two kinds draw apart;
    a unit with no seed leaves them to the global ``random``. With ``capture``
    the unit's LLM calls are tagged for internals capture under a provisional
    ``{id}/{label}/attempt_{iteration}`` root, which :func:`row_of` relabels to
    the final ``sample_id`` once it is known.
    """
    root = None
    if capture:
        unit["_capture_label"] = label
        root = unit["_internals_root"] = (
            f"{unit['id']}/{label}/attempt_{unit['iteration']}"
        )
    return run_chain(
        unit["combination"].techniques, unit["prompt"],
        gen_model=gen_model, translate_model=translate_model,
        benign_data=benign_data, prompt_dir=prompt_dir, internals_root=root,
        rng_seed=(
            None if unit.get("seed") is None
            else (unit["seed"], label, unit["id"], unit["iteration"])
        ),
    )


def pending_units(
    plan,
    prompts: pd.Series,
    position: dict,
    completed: set,
    registry: dict[str, Callable],
    live: dict[int, dict],
    **gen_kwargs,
) -> Iterator[tuple[int, Iterator]]:
    """Yield ``(i, generator)`` for each planned unit of these inputs not finished.

    Each unit is keyed by its place among the pending units, so two plan rows
    with one ``(input_id, iteration)`` stay two units, and is recorded in
    ``live`` for :func:`row_of` to read when it finishes. A unit is built only
    as the driver takes it, and carries the seed its plan row was drawn with.
    ``gen_kwargs`` go to :func:`unit_gen`.
    """
    pending = (r for r in plan if is_pending(r, position, completed))
    for i, r in enumerate(pending):
        sid = str(r["sample_id"])
        unit = {
            "id": sid,
            "prompt": prompts.iat[position[sid]],
            "combination": combination_of(r["combination"], registry),
            "iteration": int(r["iteration"]),
            "seed": (r.get("settings") or {}).get("seed"),
        }
        live[i] = unit
        yield i, unit_gen(unit, **gen_kwargs)


def _rename_capture(unit: dict, gen_model: str, sample_id: str, resolve) -> None:
    """Relabel a unit's provisional internals capture to its final sample_id.

    A no-op unless ``capture`` was on for this run, so safe to call always.
    """
    root = unit.get("_internals_root")
    if not root:
        return
    if resolve is None:
        from redact.llms.client import ModelClient
        resolve = ModelClient.create
    target = f"{unit['id']}/{unit['_capture_label']}/{sample_id}"
    resolve(gen_model).rename_capture(root, target)


def _row(unit: dict, input_row: dict, spec_version: str, fields: dict) -> dict:
    """The input's columns, the fields of a finished unit, then its iteration
    and the spec version.

    The input's own ``sample_id`` is among its columns; ``fields`` overwrites
    it with this row's own.
    """
    row = dict(input_row)
    row.update(fields)
    row["iteration"] = unit["iteration"]
    row["combination_spec_version"] = spec_version
    return row


def _fields(
    unit: dict, sample_id: str, text: str, info: str, accepted: bool, reasoning: str,
) -> dict:
    """The jailbreak fields of one row, in the column order the file keeps."""
    fn = unit["combination"]
    techniques = list(getattr(fn, "techniques", []))
    return {
        "input_id": unit["id"],
        "sample_id": sample_id,
        "jailbreak": text,
        "technique": getattr(fn, "__name__", "identity"),
        "technique_info": info,
        "complexity": sum(getattr(t, "complexity", 0) for t in techniques),
        "num_techniques": len(techniques),
        "is_noop": text == unit["prompt"],
        "accepted": accepted,
        "reasoning": reasoning,
    }


def row_of(
    unit: dict,
    value: tuple[str, str],
    input_row: dict,
    spec_version: str,
    *,
    gen_model: str,
    resolve=None,
    accept_noop: bool = True,
) -> dict:
    """The output row of a finished unit, from its ``(text, info)``.

    ``sample_id`` is this row's own content-hash identity (of the jailbroken
    text), distinct from ``input_id`` (its origin sample). With
    ``accept_noop`` False, a text that is the unit's prompt is rejected.
    """
    text, info = value
    sample_id = content_id(text)
    _rename_capture(unit, gen_model, sample_id, resolve)
    accepted = not is_discarded(info)
    reasoning = "" if accepted else info
    if accepted and not accept_noop and text == unit["prompt"]:
        accepted, reasoning = False, "unchanged: the text is its input's"
    fields = _fields(unit, sample_id, text, info, accepted, reasoning)
    return _row(unit, input_row, spec_version, fields)


def error_row_of(
    unit: dict,
    exc: Exception,
    input_row: dict,
    spec_version: str,
    *,
    gen_model: str,
    resolve=None,
    label: str = "jailbreak",
) -> dict:
    """The output row of a unit whose chain raised — the original kept.

    ``sample_id`` is this row's own, ``content_id`` of ``label`` and its unit
    key, so an error row never collides with its input row, the other kind's
    error row or any other row.
    """
    sample_id = content_id(f"{label}:{unit['id']}:{unit['iteration']}")
    _rename_capture(unit, gen_model, sample_id, resolve)
    # text is the unchanged prompt, so _fields sets is_noop True.
    fields = _fields(unit, sample_id, unit["prompt"], f"ERROR: {exc}", False, str(exc))
    return _row(unit, input_row, spec_version, fields)
