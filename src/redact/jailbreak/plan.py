"""The plan of a run: one combination per input and iteration, frozen in the
manifest, and read back.

A manifest row is ``{sample_id, iteration, combination, settings, category,
entry_type, status}``: ``combination`` names the techniques in the order they
apply, ``settings`` what the sampler was given, with the run's ``seed``. The
plan file beside the manifest holds what the whole plan was made with.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from redact.dataset import Manifest, row_id
from redact.dataset.dealing import Dealer, stable_seed
from redact.dataset.planfile import coverage_path, plan_record, write_plan

from .chain import combine_techniques
from .pool import TRANSLATION
from .sampling import sample_combination
from .technique import FIRST_VERSION, applies, load_spec

logger = logging.getLogger(__name__)

# A layer, or the request obfuscation step, whose picks fall this far or
# further below its target is logged.
_SHORT = 2


def _observe(dealer: Dealer, combined: Callable) -> None:
    """Count in ``dealer`` what one unit's draw ``combined`` holds: each
    technique's first family, the technique itself (a member of that family)
    and its layer, and the request obfuscation when it has one."""
    for fn in getattr(combined, "techniques", []):
        families = getattr(fn, "families", None) or []
        if families:
            dealer.counts[families[0]] += 1
        dealer.members[fn.__name__] += 1
        layer = getattr(fn, "layer", None)
        if layer is not None:
            dealer.included[layer] += 1
    if getattr(combined, "request_obfuscation", None) is not None:
        dealer.included["request_obfuscation"] += 1


def _names(combined: Callable) -> list[str]:
    return [getattr(t, "__name__", "") for t in getattr(combined, "techniques", [])]


def assign_combination(
    sample_id: str,
    pool: list[Callable],
    *,
    seed: int = 42,
    iteration: int = 0,
    used: tuple | list | set = (),
    max_resamples: int = 8,
    dealer=None,
    **sample_kwargs,
) -> list[str]:
    """The names of the combination one unit is assigned, in applied order.

    ``pool`` is what the unit may draw from: :func:`plan_run` passes the
    techniques that apply to the unit's input. The draw is seeded by
    ``(seed, sample_id, iteration)``, so it does not depend on the order or
    the chunking of the inputs. It is drawn again, up
    to ``max_resamples`` times, while it equals a combination in ``used`` (the
    sample's earlier iterations); the last draw is kept if none differs. The
    first draw is dealt from ``dealer``; a resample draws uniformly, so it can
    escape a collision the deal would otherwise force on a small pool. The
    kept draw is counted in ``dealer`` (its families, techniques, layers and
    request obfuscation), so the group's next unit is dealt after it.
    ``sample_kwargs`` go to :func:`sample_combination`. An empty list is the
    identity.
    """
    used_set = {tuple(c) for c in used}
    base = stable_seed(seed, sample_id, iteration)
    kept = None
    for attempt in range(max_resamples):
        rng = random.Random(base + attempt)
        kept = sample_combination(
            rng, pool, dealer=(dealer if attempt == 0 else None), **sample_kwargs)
        if tuple(_names(kept)) not in used_set:
            break
    if kept is None:
        return []
    if dealer is not None:
        _observe(dealer, kept)
    return _names(kept)


def combination_of(names: list[str], registry: dict[str, Callable]) -> Callable:
    """The combined technique a manifest row names, in the order it names them.

    Raises:
        ValueError: A name no registered technique has, as in a plan made
            before a technique was renamed or removed.
    """
    for name in names:
        if name not in registry:
            raise ValueError(
                f"The plan names the technique {name!r}, which no registered "
                "technique implements; run with resume=False to plan again.")
    return combine_techniques(*(registry[n] for n in names), sort_by_hierarchy=False)


def plan_run(
    inputs: pd.DataFrame,
    pool: list,
    *,
    manifest_path,
    seed: int = 42,
    iterations: int = 1,
    settings_per_iteration: list[dict] | None = None,
    sample_kwargs: dict | None = None,
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
    text_col: str = "prompt",
    id_col: str = "sample_id",
    verbose: bool = True,
) -> Path:
    """Write the plan of every input and iteration to ``manifest_path``.

    A unit's ``sample_id`` is the input's ``id_col`` when it has one, else the
    content id of ``text_col``. Each iteration's sampler settings are
    ``sample_kwargs`` with ``settings_per_iteration[iteration]`` merged over
    them, key by key.

    Each unit draws from the techniques of ``pool`` that apply to its input
    (:func:`~redact.jailbreak.technique.applies`, read off the input's
    ``text_col`` and ``entry_type``), in pool order: the whole pool when
    every technique applies, so the draw is the one the pool gives. A unit
    no technique of a layer applies to takes no pick there, and the dealer,
    which counts what each unit drew, gives the layer to the group's next
    unit that can take it; a unit no technique applies to gets the empty
    combination.

    The units of one ``(category, entry_type, iteration)`` group are dealt in
    ``sample_id`` order from a shared
    :class:`~redact.dataset.dealing.Dealer`, so each layer appears at its rate
    (``layer_rates`` over the spec's), each family at its weight
    (``family_weights``, equal by default) and each technique of a family as
    often as the others the unit may take, evenly across the group and
    independent of the input order; a row's ``settings`` record the weights
    when given. The plan file (:func:`plan_inputs`) is written beside the
    manifest first, and a coverage table after it; with ``verbose``, one line
    then says what the plan costs in model calls (:func:`log_model_calls`).
    Returns ``manifest_path``.
    """
    base_kwargs = dict(sample_kwargs or {})
    if layer_rates is not None:
        base_kwargs = {"sampling_probs": dict(layer_rates), **base_kwargs}
    manifest_path = Path(manifest_path)
    planned = 0

    # Each (category, entry_type) subgroup's input rows, dealt in sample_id
    # order so the plan does not depend on the order the inputs arrived in; a
    # row keeps its own dedup history across its iterations.
    subgroups: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    texts: list[str] = []
    for idx, (_, row) in enumerate(inputs.iterrows()):
        sid = row_id(row, text_col, id_col)
        key = (str(row.get("category", "")), str(row.get("entry_type", "")))
        subgroups[key].append((sid, idx))
        texts.append(str(row.get(text_col, "")))
    for members in subgroups.values():
        members.sort()
    used_by_row: dict[int, list[tuple]] = defaultdict(list)
    recorded = {"family_weights": dict(family_weights)} if family_weights else {}

    # Yielded to the manifest one at a time: the plan is never held whole.
    def units():
        nonlocal planned
        for it in range(iterations):
            settings = (
                {**base_kwargs, **settings_per_iteration[it]}
                if settings_per_iteration is not None
                else base_kwargs
            )
            for key in sorted(subgroups):
                category, entry_type = key
                dealer = Dealer(family_weights, key=(seed, category, entry_type, it))
                for sid, idx in subgroups[key]:
                    dealer.position += 1
                    candidates = [fn for fn in pool
                                  if applies(fn, texts[idx], entry_type)]
                    names = assign_combination(
                        sid, candidates, seed=seed, iteration=it,
                        used=used_by_row[idx], dealer=dealer, **settings,
                    )
                    used_by_row[idx].append(tuple(names))
                    planned += 1
                    yield {
                        "sample_id": sid,
                        "iteration": it,
                        "combination": names,
                        "settings": {**settings, "seed": seed, **recorded},
                        "category": category,
                        "entry_type": entry_type,
                        "status": "planned",
                    }
                _warn_short(dealer, it, manifest_path)

    rounds = [{**(sample_kwargs or {}),
               **(settings_per_iteration[it] if settings_per_iteration else {})}
              for it in range(iterations)]
    write_plan(manifest_path, plan_inputs(
        pool, seed=seed, rounds=rounds, family_weights=family_weights,
        layer_rates=layer_rates))
    Manifest(manifest_path).write(units())
    coverage(manifest_path, pool)

    if verbose:
        logger.info("Planned %d units (%d samples x %d) -> %s",
                    planned, len(inputs), iterations, manifest_path)
        log_model_calls(manifest_path, pool)
    return manifest_path


def log_model_calls(manifest_path: Path, pool: list[Callable]) -> None:
    """Log in one INFO line what the planned units cost in model calls.

    A unit counts when it draws a technique that calls a model
    (``requires_llm``), and its techniques' spec ``calls`` add up, low to
    high; the translation family's are also given apart, since they go to
    the translation model (a machine translation's judge aside). A technique
    whose spec gives no ``calls`` (cognitive, persona) adds none, and the
    line counts the units that draw one.
    """
    by_name = {fn.__name__: fn for fn in pool}
    units = calling = uncounted = 0
    calls, translation = [0, 0], [0, 0]
    for row in _manifest_rows(Path(manifest_path)):
        units += 1
        fns = [by_name[name] for name in row.get("combination") or []
               if getattr(by_name.get(name), "requires_llm", False)]
        calling += bool(fns)
        uncounted += any(fn.calls is None for fn in fns)
        for fn in fns:
            low, high = fn.calls or (0, 0)
            calls[0] += low
            calls[1] += high
            if TRANSLATION in fn.families:
                translation[0] += low
                translation[1] += high
    logger.info(
        "Model calls: %d of %d planned units call a model, %d to %d calls "
        "(translation %d to %d, from the spec's calls); units drawing a technique "
        "with no calls, not counted: %d.", calling, units, *calls, *translation,
        uncounted)


def _warn_short(dealer: Dealer, iteration: int, manifest_path: Path) -> None:
    """Log each layer of a dealt group, and the request obfuscation step,
    whose picks fell two or more short of its target, as when no technique
    of it fits a unit's other picks."""
    category, entry_type = dealer.key[1:3]
    for name, target in dealer.targets.items():
        if target - dealer.included[name] >= _SHORT:
            what = ("the request obfuscation step" if name == "request_obfuscation"
                    else f"the {name} layer")
            logger.warning(
                "Plan: group (%s, %s), iteration %d: %s has %d of its target %d; "
                "the pool or the compatibility rules left the rest no pick (see "
                "%s).", category, entry_type, iteration, what,
                dealer.included[name], target, coverage_path(manifest_path).name)


def plan_inputs(
    pool: list[Callable],
    *,
    seed: int,
    rounds: list[dict],
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
) -> dict:
    """What a plan is made with, its plan file's record: ``seed``, ``rounds``,
    the ``pool``'s technique names sorted, ``family_weights``,
    ``layer_rates``, and ``spec_version``, the spec's version, which dates
    the pool: a resume reads off it what the library added or retired since
    (:func:`redact.jailbreak.pipeline.run_augmentation`)."""
    return plan_record(
        seed=seed, rounds=rounds, pool=sorted(fn.__name__ for fn in pool),
        family_weights=family_weights, layer_rates=layer_rates,
        spec_version=load_spec().get("version", FIRST_VERSION))


def _manifest_rows(manifest_path: Path):
    """The planned rows read straight off the file (bad lines skipped)."""
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def coverage(manifest_path, pool: list[Callable]) -> Path | None:
    """Write ``<out>.coverage.csv`` beside the manifest and log a summary.

    One row per layer, family and technique drawn, with its count per category,
    entry type and iteration, plus a zero row for every pool technique never
    drawn, so the file records the whole pool. Returns the coverage path, or
    ``None`` when the manifest does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    out = coverage_path(manifest_path)

    by_name = {fn.__name__: fn for fn in pool}
    counts: dict[str, Counter] = {
        "layer": Counter(), "family": Counter(), "technique": Counter()}
    drawn: set[str] = set()
    for row in _manifest_rows(manifest_path):
        category = str(row.get("category", ""))
        entry_type = str(row.get("entry_type", ""))
        iteration = row.get("iteration", "")
        for name in row.get("combination", []) or []:
            drawn.add(name)
            fn = by_name.get(name)
            counts["technique"][(name, category, entry_type, iteration)] += 1
            families = getattr(fn, "families", None) or []
            if families:
                counts["family"][(families[0], category, entry_type, iteration)] += 1
            layer = getattr(fn, "layer", None)
            if layer is not None:
                counts["layer"][(layer, category, entry_type, iteration)] += 1

    rows = []
    for level in ("layer", "family", "technique"):
        for key in sorted(counts[level], key=lambda k: tuple(map(str, k))):
            name, category, entry_type, iteration = key
            rows.append({"level": level, "name": name, "category": category,
                         "entry_type": entry_type, "iteration": iteration,
                         "count": counts[level][key]})
    for fn in pool:
        if fn.__name__ not in drawn:
            rows.append({"level": "technique", "name": fn.__name__, "category": "",
                         "entry_type": "", "iteration": "", "count": 0})

    pd.DataFrame(rows, columns=[
        "level", "name", "category", "entry_type", "iteration", "count"
    ]).to_csv(out, index=False)
    logger.info("Coverage: %d of %d pool techniques drawn -> %s",
                len(drawn), len(pool), out)
    return out


def default_escalation_schedule() -> list[dict]:
    """The built-in four rounds, each one round's sampler settings.

    Rounds one and two draw exactly one and two techniques; rounds three and
    four draw by budget, the fourth with the larger one. A unit holds a
    hacking or a manipulation technique, never both, so the two rates of a
    round sum to at most 1. It is what an
    adversarial kind's ``"rounds": "escalation"`` names; each round is merged
    over the kind's budget settings.
    """
    return [
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        {
            "max_complexity": 6,
            "max_obfuscations": 2,
            "sampling_probs": {
                "hacking": 0.7, "manipulation": 0.3,
                "requests": 0.8, "request_obfuscation": 0.5,
            },
            "allow_request_obfuscation": True,
        },
        {
            "max_complexity": 9,
            "max_obfuscations": 3,
            "sampling_probs": {
                "hacking": 0.9, "manipulation": 0.1,
                "requests": 0.9, "request_obfuscation": 0.7,
            },
            "allow_request_obfuscation": True,
        },
    ]
