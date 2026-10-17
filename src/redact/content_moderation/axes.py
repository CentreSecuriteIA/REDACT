"""The generation axes: what each input sample and each response is asked to
look like, one spec per call, dealt evenly per ``(category, entry_type)``.

The table is ``configs/generation_axes.json``, a config folder's when it has
one, with one section per stage (``inputs``, ``outputs``). An axis holds a
``label``, its ``dimension`` (a generation tag of ``configs/dimensions.json``)
and ``values``, each with the ``label`` and ``hint`` the prompt shows, a
``weight`` (its share within the axis; 0 leaves it out) and optionally
``not_with`` (values of other axes it is never dealt with); or it holds
``from``, a field of the unit shown as it is. A key starting with ``_`` is a
comment.
"""

import csv
import json
import logging
import math
import random
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from numbers import Real
from pathlib import Path

from ..config_overlay import config_file
from ..dataset import Ledger, Manifest
from ..dataset.dealing import Dealer, stable_seed
from ..dataset.planfile import coverage_path
from ..dimensions import check_tags

logger = logging.getLogger(__name__)

#: The stages with axes, each a section of the table.
STAGES = ("inputs", "outputs")

# The fields of a unit an axis may show with ``from``.
_FIELDS = ("subcategory",)

_COVERAGE_COLUMNS = ("axis", "value", "category", "entry_type", "planned", "accepted")

Group = tuple[str, str]     # (category, entry_type)
Unit = tuple[Group, tuple[str, int]]    # (group, (entry, iteration))


def _items(mapping: dict) -> list:
    """The entries of ``mapping`` that are not comments."""
    return [(key, value) for key, value in mapping.items() if not key.startswith("_")]


def _table(config_dir: str | Path | None = None) -> dict:
    """Each stage's section of the table, ``config_dir``'s when it has one,
    checked against that folder's dimensions."""
    path = config_file("generation_axes.json", config_dir)
    table = json.loads(path.read_text(encoding="utf-8"))
    return {stage: _checked(f"{path}: {stage}", table.get(stage), config_dir)
            for stage in STAGES}


def _checked(where: str, section, config_dir: str | Path | None = None) -> dict:
    """``section`` without comments, each axis and value checked, the
    dimensions against the table of ``config_dir``.

    Raises:
        ValueError: A key, a name, a weight or a dimension the table may not
            hold.
    """
    if not isinstance(section, dict) or not _items(section):
        raise ValueError(f"{where} must be an object of axis: axis.")
    axes = {}
    for name, entry in _items(section):
        at = f"{where}.{name}"
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            raise ValueError(f"{at} must be an object with a label.")
        entry = dict(_items(entry))
        if "from" in entry:
            if entry["from"] not in _FIELDS or set(entry) - {"dimension"} != {
                    "label", "from"}:
                raise ValueError(
                    f"{at}: an axis with from holds only label, dimension and "
                    f"from, one of {list(_FIELDS)}.")
            axes[name] = entry
            continue
        values = entry.get("values")
        if not isinstance(values, dict) or not _items(values):
            raise ValueError(f"{at} must hold values (or from).")
        entry["values"] = {}
        for value, spec in _items(values):
            if not isinstance(spec, dict) or not all(
                    isinstance(spec.get(key), str) for key in ("label", "hint")):
                raise ValueError(f"{at}.{value} must have a label and a hint.")
            spec = dict(_items(spec))
            _check_weight(f"{at}.{value}", spec.get("weight"))
            spec.setdefault("not_with", {})
            entry["values"][value] = spec
        _check_positive(at, entry)
        axes[name] = entry
    for name, entry in axes.items():
        for value, spec in entry.get("values", {}).items():
            for other, names in spec["not_with"].items():
                known = axes.get(other, {}).get("values", {})
                if other == name or not isinstance(names, list) or not names or any(
                        n not in known for n in names):
                    raise ValueError(
                        f"{where}.{name}.{value}.not_with: {other!r} must be another "
                        "axis with values, and its list must name them: "
                        f"{list(known)}.")
    check_tags(where, {name: entry.get("dimension") for name, entry in axes.items()},
               "generation", config_dir)
    return axes


def _check_weight(at: str, weight) -> None:
    if isinstance(weight, bool) or not isinstance(weight, Real) or weight < 0:
        raise ValueError(
            f"{at}: the weight must be a number of at least 0, got {weight!r}.")


def _check_positive(at: str, entry: dict) -> None:
    if not any(spec["weight"] > 0 for spec in entry["values"].values()):
        raise ValueError(
            f"{at}: every weight is 0, so no value can be dealt; give one a weight, "
            "or write the axis as null to leave it out.")


def resolve(
    stage: str, given: dict | bool | None = None, *,
    config_dir: str | Path | None = None,
) -> dict | None:
    """The axes ``stage`` deals: its section of the table (``config_dir``'s
    when it has one) with ``given`` over it.

    ``given`` is ``None`` or ``True`` for the table's axes, ``False`` for
    none, or ``{axis: {value: weight}}``, an axis as ``None`` to leave it out.
    A value not named keeps its weight.

    Returns:
        The axes, in the table's order; ``None`` when there are none, or no
        axis with values is left.

    Raises:
        ValueError: An unknown stage, axis or value, or a weight that is not
            a number of at least 0. The message names the known ones.
    """
    if stage not in STAGES:
        raise ValueError(
            f"The stage {stage!r} has no generation axes; {list(STAGES)} do.")
    if given is False:
        return None
    axes = _table(config_dir)[stage]
    where = f"{stage}.axes"
    if given is None or given is True:
        return axes
    if not isinstance(given, dict):
        raise ValueError(
            f"{where} must be an object of axis: {{value: weight}}, true or false, "
            f"got {given!r}.")
    for name, weights in _items(given):
        if name not in axes:
            raise ValueError(
                f"{where} names no axis {name!r}; the axes are {list(axes)}.")
        if weights is None:
            del axes[name]
            continue
        entry = axes[name]
        if "values" not in entry:
            raise ValueError(
                f"{where}.{name} takes no weights: it shows the unit's "
                f"{entry['from']}; write it as null to leave it out.")
        if not isinstance(weights, dict):
            raise ValueError(
                f"{where}.{name} must be an object of value: weight, or null, "
                f"got {weights!r}.")
        for value, weight in _items(weights):
            if value not in entry["values"]:
                raise ValueError(
                    f"{where}.{name} names no value {value!r}; the values are "
                    f"{list(entry['values'])}.")
            _check_weight(f"{where}.{name}.{value}", weight)
            entry["values"][value]["weight"] = weight
        _check_positive(f"{where}.{name}", entry)
    if not any("values" in entry for entry in axes.values()):
        return None
    return axes


def varied(axes: dict | None) -> dict[str, str]:
    """The dimension of each of ``axes`` that varies from sample to sample:
    one with ``from``, or with two or more values of weight above 0. An axis
    dealt one value fixes its dimension and varies nothing."""
    return {name: entry["dimension"] for name, entry in (axes or {}).items()
            if "from" in entry
            or sum(value["weight"] > 0 for value in entry["values"].values()) > 1}


def _clashes(axes: dict, spec: dict, axis: str, value: str) -> bool:
    """Whether ``value`` of ``axis`` and a value already in ``spec`` name each
    other in ``not_with``."""
    own = axes[axis]["values"][value]["not_with"]
    if any(spec.get(other) in names for other, names in own.items()):
        return True
    return any(value in axes[other]["values"][picked]["not_with"].get(axis, ())
               for other, picked in spec.items())


def _shares(entry: dict) -> dict[str, float]:
    """Each value of an axis dealt at all, as its share of the axis's weight."""
    weights = {value: v["weight"] for value, v in entry["values"].items()
               if v["weight"] > 0}
    total = sum(weights.values())
    return {value: weight / total for value, weight in weights.items()}


def _one_spec(
    axes: dict, dealer: Dealer, rng: random.Random, taken: dict[str, Counter],
    siblings: int,
) -> dict:
    """A value for each axis with values, in the table's order.

    ``taken`` counts the values the unit's earlier siblings got; an entry of
    ``siblings`` units takes a value at most ceil(share * siblings) times.
    """
    spec: dict = {}
    for axis, entry in axes.items():
        if "values" not in entry:
            continue
        shares = _shares(entry)
        fitting = [value for value in shares
                   if not _clashes(axes, spec, axis, value)] or list(shares)
        room = [value for value in fitting
                if taken[axis][value] < math.ceil(shares[value] * siblings - 1e-9)]
        _, value = dealer.pick(rng, [(axis, value) for value in room or fitting])
        dealer.counts[(axis, value)] += 1
        spec[axis] = value
    return spec


def deal_specs(axes: dict | None, units: Sequence[Unit], *, seed: int) -> list[dict]:
    """One spec for each unit, dealt per group.

    ``units`` are ``(group, (entry, iteration))`` pairs, an entry's units its
    siblings. The units of a group are dealt in ``(entry, iteration)`` order
    from one dealer: each value goes to the unit with the lowest count over
    weight so far, ties broken by the unit's rng, seeded by ``(seed, *group,
    entry, iteration)``, so the plan does not depend on the units' order. An
    entry of k units takes a value at most ceil(share * k) times, so its
    siblings differ on every axis none of whose values has more than 1/k of
    the weight.

    Returns:
        Each unit's spec, in the order of ``units``; empty ones without
        ``axes``.
    """
    if not axes:
        return [{} for _ in units]
    weights = {(axis, value): v["weight"] for axis, entry in axes.items()
               for value, v in entry.get("values", {}).items()}
    siblings = Counter((group, entry) for group, (entry, _) in units)
    taken: dict[tuple, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    dealers: dict[Group, Dealer] = {}
    specs: list[dict] = [{} for _ in units]
    for i in sorted(range(len(units)), key=lambda i: units[i]):
        group, (entry, iteration) = units[i]
        dealer = dealers.setdefault(group, Dealer(weights, key=(seed, *group)))
        rng = random.Random(stable_seed(seed, *group, entry, iteration))
        own = taken[(group, entry)]
        specs[i] = _one_spec(axes, dealer, rng, own, siblings[(group, entry)])
        for axis, value in specs[i].items():
            own[axis][value] += 1
    return specs


def _lines(axes: dict | None, spec: dict, fields: dict) -> list[str]:
    lines = []
    for axis, entry in (axes or {}).items():
        if "from" in entry:
            if fields.get(entry["from"]):
                lines.append(f"{entry['label']}: {fields[entry['from']]}")
        elif axis in spec:
            value = entry["values"][spec[axis]]
            hint = f" — {value['hint']}" if value["hint"] else ""
            lines.append(f"{entry['label']}: {value['label']}{hint}")
    return lines


def render_spec(axes: dict | None, spec: dict, fields: dict) -> str:
    """One spec as one line per axis, in the table's words; ``fields`` gives
    an axis with ``from`` its value, and an empty one has no line."""
    return "\n".join(_lines(axes, spec, fields))


def spec_columns(axes: dict | None, spec: dict, *, prefix: str = "") -> dict:
    """A row's columns of ``spec``: one per axis with values, named
    ``prefix`` + the axis."""
    return {prefix + axis: spec.get(axis, "")
            for axis, entry in (axes or {}).items() if "values" in entry}


def write_coverage(
    path: str | Path,
    axes: dict,
    planned: Iterable[tuple[Group, dict]],
    accepted: Iterable[tuple[Group, dict]],
) -> Path:
    """Write the planned and accepted count of each axis value per group.

    ``planned`` and ``accepted`` give one ``(group, spec)`` per sample. Every
    value of the table has a row in every group seen, at 0 when never dealt;
    a value the table no longer has keeps its counts, after them.
    """
    counts = {"planned": Counter(), "accepted": Counter()}
    for kind, pairs in (("planned", planned), ("accepted", accepted)):
        for group, spec in pairs:
            for axis, value in spec.items():
                counts[kind][(tuple(group), axis, value)] += 1
    groups = sorted({key[0] for kind in counts.values() for key in kind})
    keys = [(group, axis, value) for group in groups
            for axis, entry in axes.items() for value in entry.get("values", {})]
    listed = set(keys)
    keys += sorted({key for kind in counts.values() for key in kind} - listed)
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_COVERAGE_COLUMNS)
        for (category, entry_type), axis, value in keys:
            key = ((category, entry_type), axis, value)
            writer.writerow([axis, value, category, entry_type,
                             counts["planned"][key], counts["accepted"][key]])
    return path


def _group(row: dict) -> Group:
    return str(row.get("category", "")), str(row.get("entry_type", ""))


def write_stage_coverage(
    axes: dict | None, manifest: Manifest, ledger: Ledger,
) -> Path | None:
    """Write the coverage table beside ``manifest``: each row's ``spec``
    planned, and accepted when the ack of its unit in ``ledger`` says
    ``accepted``, by the row's ``category`` and ``entry_type``. No row of the
    output is read.

    Without ``axes`` the table is removed, so it never describes a plan the
    manifest does not hold.
    """
    path = coverage_path(manifest.path)
    if not axes:
        path.unlink(missing_ok=True)
        return None
    accepted = {ledger.key(ack) for ack in ledger.records()
                if ack.get("accepted") is True}
    planned = [(_group(row), row["spec"], ledger.key(row))
               for row in manifest if row.get("spec")]
    write_coverage(path, axes, ((group, spec) for group, spec, _ in planned),
                   ((group, spec) for group, spec, key in planned if key in accepted))
    logger.info("Coverage of the generation axes -> %s", path)
    return path


def warn_unshown(prompt_config: dict, placeholder: str, name: str) -> None:
    """Warn when the prompt ``name`` has no ``{placeholder}``, as an overlay
    written before the axes: the specs are dealt and recorded, but the model
    is not shown them."""
    fields: set[str] = set()
    for text in (prompt_config.get("system_prompt"), prompt_config.get("template")):
        try:
            fields.update(f for _, f, _, _ in string.Formatter().parse(text or "") if f)
        except ValueError:      # malformed: the render says so
            return
    if placeholder not in fields:
        logger.warning(
            "The prompt %s has no {%s}: each unit's spec is dealt and recorded, "
            "and the model is not shown it. Add {%s} to its template.",
            name, placeholder, placeholder)
