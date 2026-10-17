"""The generation axes: what each input sample and each response is asked to
look like, one spec per sample, dealt evenly per ``(category, entry_type)``.

The packaged table is ``configs/generation_axes.json``, one section per stage
(``inputs``, ``outputs``). An axis holds a ``label``, its ``dimension`` (a
generation tag of ``configs/dimensions.json``) and ``values``, each with the
``label`` and ``hint`` the prompt shows, a ``weight`` (its share within the
axis; 0 leaves it out) and optionally ``not_with`` (values of other axes it is
never dealt with); or it holds ``from``, a field of the unit shown as it is.
A key starting with ``_`` is a comment.
"""

import copy
import csv
import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from numbers import Real
from pathlib import Path

from .. import paths
from ..dataset.dealing import Dealer, stable_seed
from ..dimensions import check_tags

#: The stages with axes, each a section of the table.
STAGES = ("inputs", "outputs")

# The fields of a unit an axis may show with ``from``.
_FIELDS = ("subcategory",)

_COVERAGE_COLUMNS = ("axis", "value", "category", "entry_type", "planned", "accepted")

Group = tuple[str, str]     # (category, entry_type)


def _items(mapping: dict) -> list:
    """The entries of ``mapping`` that are not comments."""
    return [(key, value) for key, value in mapping.items() if not key.startswith("_")]


@lru_cache(maxsize=1)
def _packaged() -> dict:
    path = paths.configs_dir() / "generation_axes.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    sections = {}
    for stage in STAGES:
        where = f"{path.name}: {stage}"
        sections[stage] = _checked(where, table.get(stage))
    return sections


def _checked(where: str, section) -> dict:
    """``section`` without comments, each axis and value checked.

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
               "generation")
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


def resolve(stage: str, given: dict | bool | None = None) -> dict | None:
    """The axes ``stage`` deals: its packaged section with ``given`` over it.

    ``given`` is ``None`` or ``True`` for the packaged axes, ``False`` for
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
    axes = copy.deepcopy(_packaged()[stage])
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


def _clashes(axes: dict, spec: dict, axis: str, value: str) -> bool:
    """Whether ``value`` of ``axis`` and a value already in ``spec`` name each
    other in ``not_with``."""
    own = axes[axis]["values"][value]["not_with"]
    if any(spec.get(other) in names for other, names in own.items()):
        return True
    return any(value in axes[other]["values"][picked]["not_with"].get(axis, ())
               for other, picked in spec.items())


def _one_spec(axes: dict, dealer: Dealer, rng: random.Random) -> dict:
    """A value for each axis with values, in the table's order."""
    spec: dict = {}
    for axis, entry in axes.items():
        if "values" not in entry:
            continue
        dealt = [value for value, v in entry["values"].items() if v["weight"] > 0]
        fitting = [value for value in dealt if not _clashes(axes, spec, axis, value)]
        _, value = dealer.pick(rng, [(axis, value) for value in fitting or dealt])
        dealer.counts[(axis, value)] += 1
        spec[axis] = value
    return spec


def deal(
    axes: dict | None, units: Sequence[tuple[Group, str]], count: int, *, salt: str,
) -> list[list[dict]]:
    """``count`` specs for each unit, dealt per group.

    ``units`` are ``(group, name)`` pairs. The units of a group are dealt in
    ``name`` order from one dealer: each value goes to the specs with the
    lowest count over weight so far, ties broken by the unit's rng, seeded by
    ``(salt, *group, name)``. So the plan does not depend on the units' order.

    Returns:
        Each unit's specs, in the order of ``units``; empty lists without
        ``axes``.
    """
    if not axes:
        return [[] for _ in units]
    weights = {(axis, value): v["weight"] for axis, entry in axes.items()
               for value, v in entry.get("values", {}).items()}
    dealers: dict[Group, Dealer] = {}
    specs: list[list[dict]] = [[] for _ in units]
    for i in sorted(range(len(units)), key=lambda i: units[i]):
        group, name = units[i]
        dealer = dealers.setdefault(group, Dealer(weights, key=group))
        rng = random.Random(stable_seed(salt, *group, name))
        specs[i] = [_one_spec(axes, dealer, rng) for _ in range(count)]
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


def render_specs(axes: dict | None, specs: list[dict], fields: dict) -> str:
    """The specs of one call as a numbered list, a spec's lines under its number."""
    blocks = []
    for n, spec in enumerate(specs, 1):
        lines = _lines(axes, spec, fields)
        if lines:
            prefix = f"{n}. "
            blocks.append(prefix + ("\n" + " " * len(prefix)).join(lines))
    return "\n".join(blocks)


def columns(axes: dict | None, spec: dict, *, prefix: str) -> dict:
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
