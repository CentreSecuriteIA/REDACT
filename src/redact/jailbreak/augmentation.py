"""The augmentation section of a run: its two kinds, their defaults, and what a
section says.

A section is ``{"adversarial": {...}, "natural": {...}}``. A kind given
overlays its default from ``configs/jailbreak/augmentations.json`` (a config
folder's, given one, see :mod:`redact.config_overlay`) key by key:
a value given replaces the default's whole, except ``layers``, ``families``
and ``techniques``, whose ``allow`` and ``deny`` lists replace the default's
one by one. A kind given as ``None`` is off; a kind left out is the default.
A key starting with ``_``, in the section or in a kind, is a comment and is
skipped. ``translation`` alone decides whether a kind draws the translation
family; the lists may narrow it to the languages they name. A kind draws
only the techniques whose ``kinds`` in ``combination_spec.json`` hold it, so
the natural kind draws only what the spec marks natural. A layer the spec
has off by default (``layers_off_by_default``: ``rewrite``) is denied by
every kind whose ``layers.allow`` does not name it, whatever its
``layers.deny`` says.
"""

import copy
import json
import logging
from functools import lru_cache
from numbers import Real
from pathlib import Path

from redact.config_overlay import config_file
from redact.types import ALL_ENTRY_TYPES

from .plan import UNIT_HEADROOM, default_escalation_schedule
from .pool import (
    DIALECT,
    TRANSLATION,
    TRANSLATIONESE,
    all_techniques,
    dialects,
    languages,
    technique_pool,
    translationese_names,
)
from .technique import DEFAULT_KINDS, KINDS, family_kinds, load_spec

logger = logging.getLogger(__name__)

_COMMON_KEYS = ("entry_types", "translation", "layers", "families", "techniques",
                "pure_only", "family_weights", "iterations")
#: The group budget's keys: set only on an adversarial kind, and written to
#: its rounds only when set (``budget`` only when ``"group"``).
_GROUP_BUDGET_KEYS = ("budget", "max_unit_complexity")
_KIND_KEYS = {
    "adversarial": (*_COMMON_KEYS, "rounds", "max_complexity", "max_obfuscations",
                    "layer_rates", "allow_request_obfuscation", *_GROUP_BUDGET_KEYS),
    "natural": (*_COMMON_KEYS, "exact_techniques"),
}
_ROUND_KEYS = ("exact_techniques", "max_complexity", "max_obfuscations",
               "layer_rates", "allow_request_obfuscation")
_RATE_KEYS = ("rewrite", "hacking", "manipulation", "requests", "request_obfuscation")
#: Not the apply order (the spec's ``layer_order``, manipulation last): the
#: order a round writes its ``include_<layer>`` keys in, so the key order of
#: every plan file's ``rounds``, which ``tests/fixtures/stage_files.json``
#: pins. A resume compares the rounds as values, whatever their key order.
_LAYERS = ("rewrite", "hacking", "manipulation", "obfuscation", "requests")
#: The layer the spec has off by default: a kind denies it unless its
#: ``layers.allow`` names it, and a round says ``include_rewrite`` only for a
#: kind whose pool holds a technique of the layer, so the rounds of every
#: other kind are those written before the layer existed.
_REWRITE = "rewrite"
_SPEC = "combination_spec.json"
_NATURAL_ONLY = "natural_only_families"

#: Today's keywords of ``generate_jailbreaks``, each an alias into the
#: adversarial kind for one release (:func:`apply_flat_keywords`).
FLAT_KEYWORDS = (
    "include_hacking", "include_manipulation", "include_obfuscation",
    "include_requests", "include_translation", "pure_only", "max_complexity",
    "max_obfuscations", "sampling_probs", "iterations", "settings_per_iteration",
    "entry_types",
)


@lru_cache(maxsize=8)
def _read(path: str, modified: int) -> dict:
    return _checked_file(path, json.loads(Path(path).read_text(encoding="utf-8")))


def _section_file(config_dir: str | Path | None) -> dict:
    """``augmentations.json``, checked (:func:`_checked_file`) and read again
    when the file changes."""
    path = config_file("jailbreak/augmentations.json", config_dir)
    return _read(str(path), path.stat().st_mtime_ns)


def _checked_file(path: str, data) -> dict:
    """``data``, the ``augmentations.json`` at ``path``, with each default
    kind checked as a section's kind is: a ``config_dir`` copy is read with
    no recipe to catch what it holds.

    A key the file or a kind may not hold (a typo, as ``max_complexty`` or
    ``layers.alow``) did nothing before, so a copy may hold one: it is left
    out with a WARNING that names it. A kind's ``layers``, ``families`` and
    ``techniques`` are read as a section's (:func:`_lists`), the ``allow`` a
    ``families`` or ``techniques`` leaves out ``None`` and a ``deny`` ``[]``.

    Raises:
        ValueError: A kind is not an object or lacks a key, the
            adversarial kind gives ``exact_techniques`` (it would be read as
            the natural kind, :func:`~redact.jailbreak.pool.kind_of`), a list
            is malformed, or ``natural_only_families`` is missing or names a
            family the spec does not have. The message names the file and
            the key.
    """
    known = [*KINDS, _NATURAL_ONLY]
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold an object of {known}, got {data!r}.")
    unknown = [key for key in data if key not in known and not key.startswith("_")]
    if unknown:
        logger.warning("%s has unknown key(s) %s, which are ignored; it holds %s.",
                       path, unknown, known)
    if _NATURAL_ONLY not in data:
        raise ValueError(
            f"{path} lacks {_NATURAL_ONLY}, the families the attacks draw only when "
            "their families.allow names them; write it as the library's file does, "
            "delete the copy, or write it again with scaffold_setup(..., "
            "overwrite=True), which replaces your edits.")
    only, families = data[_NATURAL_ONLY], load_spec()["families"]
    if not isinstance(only, list) or not all(name in families for name in only):
        raise ValueError(
            f"{path}: {_NATURAL_ONLY} must be a list of families of {_SPEC}, [] for "
            f"none, got {only!r}.")
    return {**{key: value for key, value in data.items() if key not in unknown},
            **{kind: _checked_default(f"{path}: {kind}", kind, data.get(kind))
               for kind in KINDS}}


def _checked_default(where: str, kind: str, value) -> dict:
    """``value``, a default kind of ``augmentations.json``, checked
    (:func:`_checked_file`)."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{where} must be an object of the kind's keys, got {value!r}; a "
            "dataset type file or a recipe switches a kind off with null.")
    value = {key: v for key, v in value.items() if not key.startswith("_")}
    if kind == "adversarial" and "exact_techniques" in value:
        raise ValueError(
            f"{where} gives exact_techniques, a key of the natural kind: the "
            "attacks would be read as natural and draw its pool. Take it out; a "
            "round of rounds may give it.")
    unknown = [key for key in value if key not in _KIND_KEYS[kind]]
    if unknown:
        logger.warning("%s has unknown key(s) %s, which are ignored; write only %s.",
                       where, unknown, list(_KIND_KEYS[kind]))
    value = {key: v for key, v in value.items() if key not in unknown}
    missing = [key for key in _KIND_KEYS[kind]
               if key not in value and key not in _GROUP_BUDGET_KEYS]
    if missing:
        raise ValueError(
            f"{where} lacks {missing}; write them as the library's file does, "
            "delete the copy, or write it again with scaffold_setup(..., "
            "overwrite=True), which replaces your edits.")
    for level in ("layers", "families", "techniques"):
        given = value[level]
        extra = [key for key in given if key not in ("allow", "deny")] if isinstance(
            given, dict) else []
        if extra:
            logger.warning(
                "%s.%s has unknown key(s) %s, which are ignored; write only "
                "['allow', 'deny'].", where, level, extra)
            given = {key: given[key] for key in ("allow", "deny") if key in given}
        lists = _lists(where, level, given)
        value[level] = lists if level != "layers" else {
            **({"allow": lists["allow"]} if "allow" in given else {}),
            "deny": lists["deny"]}
    return value


def defaults(config_dir: str | Path | None = None) -> dict:
    """The default section, both kinds, as a copy the caller may change: the
    packaged one, or ``config_dir``'s when it has the file, checked
    (:func:`_checked_file`)."""
    return copy.deepcopy({kind: _section_file(config_dir)[kind] for kind in KINDS})


def natural_only_families(config_dir: str | Path | None = None) -> list[str]:
    """The families the adversarial kind draws only when its
    ``families.allow`` names them."""
    return list(_section_file(config_dir)[_NATURAL_ONLY])


def layers_off_by_default() -> list[str]:
    """The layers a kind denies unless its ``layers.allow`` names them: the
    spec's ``layers_off_by_default``."""
    return list(load_spec().get("layers_off_by_default", []))


def resolve(
    section: dict | None, kind: str, base: dict | None = None,
    config_dir: str | Path | None = None,
) -> dict | None:
    """One kind of ``section``, overlaid on its default and checked.

    ``section`` may be ``None`` (both kinds the default) or a section already
    resolved, which comes back unchanged. The default is the packaged kind,
    or, with ``base``, ``base``'s kind resolved over it: a kind ``section``
    leaves out is then ``base``'s (off when ``base`` gives it as ``None``),
    and a kind it gives overlays ``base``'s, or the packaged one when
    ``base`` switches it off; ``config_dir``'s ``augmentations.json``
    replaces the packaged one when it has the file. ``families`` and
    ``techniques`` come back with both their lists (``allow`` ``None`` for
    all), ``layers`` with its ``deny`` and, when one was given, its
    ``allow``, and a kind with ``rounds`` with ``iterations`` set to their
    number. ``layers.deny`` holds every layer of
    :func:`layers_off_by_default` its ``layers.allow`` does not name
    (:func:`_deny_layers_off`), so the pool and the rounds read one list.
    The adversarial ``families.deny`` holds every family of
    :func:`natural_only_families` its ``families.allow`` does not name.
    An allow list names only families and techniques whose spec ``kinds``
    hold the kind (:func:`_check_kinds`), and of a layer the kind draws: a
    name of a layer off by default raises, saying to write
    ``layers.allow``, and one of a layer its ``layers.deny`` names is
    dropped with a WARNING (:func:`_check_denied_layers`).
    ``translation`` is true or false; while it is false, what the allow
    lists and ``family_weights`` name of translation is left out, with a
    WARNING (:func:`_switch_off_translation`).

    Returns:
        The full kind, or ``None`` when ``section`` switches it off.

    Raises:
        ValueError: A key, a name or a value the section may not hold. The
            message names the kind and the key and says what to write.
    """
    if kind not in KINDS:
        raise ValueError(f"No augmentation kind {kind!r}; the kinds are {list(KINDS)}.")
    section = {} if section is None else section
    if not isinstance(section, dict):
        raise ValueError(
            f"augmentations must be an object of {list(KINDS)}, got {section!r}.")
    unknown = [key for key in section if key not in KINDS and not key.startswith("_")]
    if unknown:
        raise ValueError(
            f"augmentations has unknown key(s) {unknown}; write {list(KINDS)}, "
            "each an object or null.")
    if kind in section and section[kind] is None:
        return None
    default = (resolve(base, kind, config_dir=config_dir)
               if base is not None else None)
    if base is not None and kind not in section:
        return default
    default = default or defaults(config_dir)[kind]
    given = section.get(kind, {})
    where = f"augmentations.{kind}"
    if not isinstance(given, dict):
        raise ValueError(f"{where} must be an object or null, got {given!r}.")
    given = {key: value for key, value in given.items() if not key.startswith("_")}
    budgeted = [key for key in given if key in _GROUP_BUDGET_KEYS]
    if kind == "natural" and budgeted:
        raise ValueError(
            f"{where} gives {budgeted}, which the natural kind ignores: it draws "
            "exact_techniques, a count with no complexity budget to bank. Take "
            "them out.")
    unknown = [key for key in given if key not in _KIND_KEYS[kind]]
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {unknown}; write only "
            f"{list(_KIND_KEYS[kind])}.")

    resolved = {**copy.deepcopy(default), **copy.deepcopy(given)}
    _check_entry_types(where, resolved["entry_types"])
    for level in ("layers", "families", "techniques"):
        if level in given:
            lists = _lists(where, level, given[level])
            resolved[level] = _overlay(
                where, level, default[level],
                {key: lists[key] for key in ("allow", "deny") if key in given[level]})
    if (kind == "natural" and resolved["families"]["allow"] is None
            and resolved["techniques"]["allow"] is None):
        raise ValueError(
            f"{where} needs an allow list: write families.allow or "
            "techniques.allow, as the natural kind draws only what they name.")
    if kind == "adversarial":
        families = resolved["families"]
        families["deny"] += [name for name in natural_only_families(config_dir)
                             if name not in (families["allow"] or ())
                             and name not in families["deny"]]
    _check_names(where, resolved)
    _deny_layers_off(where, resolved["layers"])
    _check_kinds(where, kind, resolved)
    _warn_retired(where, resolved)
    _switch_off_translation(where, resolved)
    _check_denied_layers(where, resolved)
    if type(resolved["pure_only"]) is not bool:
        raise ValueError(
            f"{where}.pure_only must be true or false, got {resolved['pure_only']!r}.")
    _check_count(where, "iterations", resolved["iterations"], least=1)
    _check_weights(where, resolved)
    if kind == "natural":
        _check_count(where, "exact_techniques", resolved["exact_techniques"], least=1)
        return resolved

    _check_budget(where, resolved)
    if resolved["rounds"] is not None:
        rounds = _rounds(where, resolved["rounds"])
        if "iterations" in given and given["iterations"] != len(rounds):
            raise ValueError(
                f"{where}: rounds gives {len(rounds)} rounds and iterations says "
                f"{given['iterations']}; the rounds set the number, so leave "
                "iterations out.")
        resolved["iterations"] = len(rounds)
        ceiling = resolved.get("max_unit_complexity")
        for i, r in enumerate(rounds):
            if ceiling is not None and r.get("max_complexity", 0) > ceiling:
                raise ValueError(
                    f"{where}.rounds[{i}].max_complexity is {r['max_complexity']}, "
                    f"above max_unit_complexity {ceiling}; raise "
                    "max_unit_complexity, or leave it out for each round's "
                    f"max_complexity + {UNIT_HEADROOM}.")
    _warn_rewrite_rate(where, resolved)
    return resolved


def rounds_of(resolved: dict) -> list[dict]:
    """The sampler settings of each round of a resolved kind, one dict a round.

    A natural round draws ``exact_techniques``. An adversarial round is the
    budget keys, each layer's ``include_*``, and the round's own keys from
    ``rounds`` merged over them (``"escalation"`` is
    :func:`default_escalation_schedule`), a round's ``layer_rates`` passed on
    as the sampler's ``sampling_probs``. ``include_rewrite`` is written only
    when the kind's pool holds a rewrite technique (the layer allowed, and
    drawn by its lists), where the others are always written: a round of
    any other kind, the eval type's with the layer allowed among them, is
    as it was before the layer existed (a plan kept from a kind that
    allowed the layer before its pool held a rewrite technique has no key,
    so it raises on resume, naming ``rounds``). A count round does not read
    ``include_rewrite``; its pool decides. Under ``"escalation"`` a kind
    that draws the layer carries its ``layer_rates.rewrite``, when it gives
    one, into the budget rounds' rates, which replace the kind's whole.
    ``budget`` is written
    only when ``"group"`` and
    ``max_unit_complexity`` only when given, so a kind that leaves them out
    or says ``"unit"`` has the rounds, and the plan file, it had before
    they existed; a count round carries them and ignores them.
    """
    if "exact_techniques" in resolved:
        return [{"exact_techniques": resolved["exact_techniques"]}
                for _ in range(resolved["iterations"])]
    denied = resolved["layers"]["deny"]
    base = {
        "max_complexity": resolved["max_complexity"],
        "max_obfuscations": resolved["max_obfuscations"],
        **{f"include_{layer}": layer not in denied
           for layer in _LAYERS if layer != _REWRITE},
    }
    rewrites = _REWRITE not in denied and any(
        getattr(fn, "layer", None) == _REWRITE for fn in technique_pool(resolved))
    if rewrites:
        base["include_rewrite"] = True
    if resolved["allow_request_obfuscation"]:
        base["allow_request_obfuscation"] = True
    if resolved.get("budget") == "group":
        base["budget"] = "group"
    if resolved.get("max_unit_complexity") is not None:
        base["max_unit_complexity"] = resolved["max_unit_complexity"]
    rounds = resolved["rounds"]
    if rounds is None:
        return [dict(base) for _ in range(resolved["iterations"])]
    if rounds == "escalation":
        schedule = default_escalation_schedule()
        rate = resolved["layer_rates"].get(_REWRITE)
        if rewrites and rate is not None:
            # A budget round's rates replace the kind's: carry its rewrite rate.
            schedule = [{**r, "sampling_probs": {_REWRITE: rate, **r["sampling_probs"]}}
                        if "sampling_probs" in r else r for r in schedule]
        return [{**base, **r} for r in schedule]
    return [{**base, **{("sampling_probs" if key == "layer_rates" else key): value
                        for key, value in r.items()}}
            for r in rounds]


def apply_flat_keywords(kind: dict, keywords: dict) -> dict:
    """``kind`` (adversarial, resolved) with today's flat keywords applied.

    ``include_<layer>`` denies or admits the layer, ``include_translation``
    is ``translation``; ``sampling_probs`` is ``layer_rates`` and
    ``settings_per_iteration`` is ``rounds``, each round's ``sampling_probs``
    its ``layer_rates``. A keyword that is ``None`` is not given.

    Raises:
        ValueError: ``iterations`` differs from the number of rounds of
            ``settings_per_iteration``.
    """
    kind = copy.deepcopy(kind)
    given = {key: value for key, value in keywords.items() if value is not None}
    rounds = given.get("settings_per_iteration")
    if (isinstance(rounds, list) and "iterations" in given
            and given["iterations"] != len(rounds)):
        raise ValueError(
            f"iterations is {given['iterations']} and settings_per_iteration gives "
            f"{len(rounds)} rounds; the rounds set the number, so leave iterations "
            "out.")
    for layer in _LAYERS:
        if f"include_{layer}" in given:
            kind["layers"]["deny"] = _toggled(
                kind["layers"]["deny"], layer, deny=not given[f"include_{layer}"])
    if "include_translation" in given:
        kind["translation"] = given["include_translation"]
    for key in ("pure_only", "max_complexity", "max_obfuscations", "iterations",
                "entry_types"):
        if key in given:
            kind[key] = given[key]
    if "sampling_probs" in given:
        kind["layer_rates"] = given["sampling_probs"]
    if isinstance(rounds, list):
        kind["rounds"] = [
            {("layer_rates" if key == "sampling_probs" else key): value
             for key, value in r.items()} if isinstance(r, dict) else r
            for r in rounds]
        kind["iterations"] = len(rounds)
    elif rounds is not None:
        kind["rounds"] = rounds
    return kind


def _toggled(names: list, name: str, *, deny: bool) -> list:
    """``names`` with ``name`` in it when ``deny``, else without it."""
    if deny:
        return names if name in names else [*names, name]
    return [n for n in names if n != name]


def _check_entry_types(where: str, entry_types) -> None:
    valid = [str(t.value) for t in ALL_ENTRY_TYPES]
    if entry_types is None:
        return
    if not isinstance(entry_types, list) or not entry_types:
        raise ValueError(
            f"{where}.entry_types must be a non-empty list of {valid}, or null for "
            f"all, got {entry_types!r}.")
    unknown = [t for t in entry_types if t not in valid]
    if unknown:
        raise ValueError(
            f"{where}.entry_types names {unknown}; write entry types of {valid}.")


def _lists(where: str, level: str, value) -> dict:
    """``value`` as ``{"allow": list | None, "deny": list}``.

    A ``layers.allow`` is a list, empty for none: it names the layers off by
    default to turn on. A ``families`` or ``techniques`` ``allow`` is a
    non-empty list, or ``None`` for all.
    """
    keys = ["allow", "deny"]
    if not isinstance(value, dict):
        raise ValueError(f"{where}.{level} must be an object of {keys}, got {value!r}.")
    unknown = [key for key in value if key not in keys]
    if unknown:
        raise ValueError(
            f"{where}.{level} has unknown key(s) {unknown}; write only {keys}.")
    if level == "layers":
        allow = value.get("allow", [])
        if not isinstance(allow, list):
            raise ValueError(
                f"{where}.layers.allow must be a list of the layers to turn on, "
                f"[] for none, got {allow!r}.")
    else:
        allow = value.get("allow")
        if allow is not None and (not isinstance(allow, list) or not allow):
            raise ValueError(
                f"{where}.{level}.allow must be a non-empty list, got {allow!r}; "
                "write null for all, or name what to leave out under deny.")
    out = {"allow": None if allow is None else list(allow)}
    deny = value.get("deny", [])
    if not isinstance(deny, list):
        raise ValueError(f"{where}.{level}.deny must be a list, got {deny!r}.")
    out["deny"] = list(deny)
    if out["allow"]:
        both = [name for name in out["allow"] if name in out["deny"]]
        if both:
            raise ValueError(
                f"{where}.{level} names {both} under both allow and deny; write "
                "each under one of them.")
    return out


def _overlay(where: str, level: str, default: dict, given: dict) -> dict:
    """The lists ``given`` holds over ``default``'s, one by one.

    A name given under one list leaves the default's other list, so a deny
    given alone is the default's allow less those names. A deny may empty
    the default's ``layers.allow``, which then turns no layer on; one that
    empties a families or techniques allow, which would draw nothing, raises.
    """
    out = {**default, **given}
    for key, other in (("allow", "deny"), ("deny", "allow")):
        if given.get(key) and other not in given and out.get(other):
            out[other] = [name for name in out[other] if name not in given[key]]
            if other == "allow" and not out[other] and level != "layers":
                raise ValueError(
                    f"{where}.{level}.deny leaves out every name of the default's "
                    f"allow; write {level}.allow as well.")
    return out


def _check_names(where: str, resolved: dict) -> None:
    """Every layer, family and technique named is the spec's, and every
    technique is registered."""
    spec = load_spec()
    layers = list(spec["layer_order"])
    families = sorted(spec["families"])
    in_spec = {name for name in spec["techniques"] if not name.startswith("_")}
    registered = {fn.__name__ for fn in all_techniques()}
    for level in ("allow", "deny"):
        for name in resolved["layers"].get(level) or ():
            if name not in layers:
                raise ValueError(
                    f"{where}.layers.{level} names {name!r}, which is not a layer "
                    f"of {_SPEC}; write one of {layers}.")
    for level in ("allow", "deny"):
        for name in resolved["families"][level] or ():
            if name not in families:
                raise ValueError(
                    f"{where}.families.{level} names {name!r}, which is not a "
                    f"family of {_SPEC}; write one of {families}.")
        for name in resolved["techniques"][level] or ():
            if name not in in_spec:
                raise ValueError(
                    f"{where}.techniques.{level} names {name!r}, which {_SPEC} "
                    "does not list; write a technique name from its techniques.")
            if name not in registered:
                raise ValueError(
                    f"{where}.techniques.{level} names {name!r}, which no "
                    "registered technique implements; write a registered one.")


def _deny_layers_off(where: str, layers: dict) -> None:
    """Add to ``layers["deny"]`` every layer of :func:`layers_off_by_default`
    that ``layers["allow"]`` does not name.

    Such a layer stays off whatever ``deny`` says: a kind's own
    ``layers.deny``, or a ``config_dir`` copy of ``augmentations.json`` that
    says ``"deny": []``, leaves it off, and only ``layers.allow`` turns it
    on.

    Raises:
        ValueError: ``allow`` names a layer that is on by default, or one
            ``deny`` names too.
    """
    off = layers_off_by_default()
    allow = layers.get("allow") or []
    for name in allow:
        if name not in off:
            raise ValueError(
                f"{where}.layers.allow names {name!r}, which is on already: "
                f"layers.allow turns on only the layers {_SPEC} has off by "
                f"default ({off}). Take it out of layers.allow.")
    both = [name for name in allow if name in layers["deny"]]
    if both:
        raise ValueError(
            f"{where}.layers names {both} under both allow and deny; write each "
            "under one of them.")
    layers["deny"] += [name for name in off
                       if name not in allow and name not in layers["deny"]]


def _check_kinds(where: str, kind: str, resolved: dict) -> None:
    """Every family and technique the allow lists name is one ``kind`` may
    draw: its spec ``kinds`` hold it, as :func:`technique_pool` reads them;
    and every layer ``layers.allow`` names holds one, or none at all.

    Raises:
        ValueError: An allow list names one the spec keeps from ``kind``, as
            a copy of ``augmentations.json`` made before it left the kind
            does. The message names it and the fixes: for a technique, take
            it out or add ``kind`` to its spec ``kinds``; for a family, take
            it out and list under ``techniques.allow`` those of its
            techniques ``kind`` draws. Adding ``kind`` to a family's
            ``kinds`` is not offered: they apply to every technique in it
            that sets none of its own. Or ``layers.allow`` names a layer
            whose techniques ``kind`` draws none of (the rewrite layer on
            the natural kind), which would turn nothing on.
    """
    registered = all_techniques()
    techniques = {fn.__name__: getattr(fn, "kinds", DEFAULT_KINDS)
                  for fn in registered}
    for layer in resolved["layers"].get("allow") or ():
        members = [fn.__name__ for fn in registered
                   if getattr(fn, "layer", None) == layer]
        if members and not any(kind in techniques[name] for name in members):
            raise ValueError(
                f"{where}.layers.allow names {layer!r}, none of whose techniques "
                f"the {kind} kind draws (their kinds in {_SPEC}), so it turns "
                "nothing on; take it out of layers.allow.")
    families = load_spec()["families"]
    for level in ("families", "techniques"):
        for name in resolved[level]["allow"] or ():
            if name not in (families if level == "families" else techniques):
                continue        # _check_names refuses a name nothing has
            kinds = (family_kinds(name) if level == "families"
                     else techniques[name])
            if kind in kinds:
                continue
            refused = (f"{where}.{level}.allow names {name!r}, which the {kind} "
                       f"kind does not draw: its kinds in {_SPEC} are {list(kinds)}")
            take_out = (f"Take it out of {level}.allow (a config_dir copy of "
                        f"augmentations.json made before it left the {kind} kind "
                        "still names it)")
            if level == "techniques":
                raise ValueError(
                    f'{refused}. {take_out}, or add "{kind}" to its kinds in {_SPEC}.')
            drawn = [fn.__name__ for fn in registered
                     if name in getattr(fn, "families", ())
                     and kind in techniques[fn.__name__]]
            members = (
                f"list under techniques.allow those of its techniques the {kind} "
                f"kind draws: {drawn}." if drawn else
                f"the {kind} kind draws none of its techniques (to draw one, add "
                f'"{kind}" to that technique\'s kinds in {_SPEC}).')
            raise ValueError(
                f"{refused}, and a family's kinds apply to every technique in it "
                f"that sets none of its own. {take_out}; {members}")


def _warn_retired(where: str, resolved: dict) -> None:
    """Log a WARNING naming each technique ``techniques.allow`` names that
    the spec retired (its ``deprecated`` version). The pool draws it for
    this release with a ``DeprecationWarning``
    (:func:`~redact.jailbreak.pool.technique_pool`), which Python shows only
    for code run as ``__main__``, so a run would not see it."""
    named = resolved["techniques"]["allow"] or ()
    retired = [f"{fn.__name__} ({fn.deprecated})" for fn in all_techniques()
               if getattr(fn, "deprecated", None) and fn.__name__ in named]
    if retired:
        logger.warning(
            "%s.techniques.allow names %s, which the spec retired at the version "
            "given; take them out of techniques.allow. They are drawn for this "
            "release.", where, ", ".join(retired))


def _check_denied_layers(where: str, resolved: dict) -> None:
    """Every family and technique the allow lists and ``family_weights``
    name is of a layer the kind draws.

    The pool leaves out a denied layer whatever the lists say, so a name of
    one is dropped there. For a layer :func:`layers_off_by_default` leaves
    off, that would be silent, and a kind that allows nothing else would
    draw nothing: it raises, saying to write ``layers.allow``. For a layer
    the kind's ``layers.deny`` names, as a run that denies a layer of the
    eval type's lists does, the names are dropped with a WARNING. A
    ``family_weights`` entry of a denied layer raises either way, as one of
    any family the kind does not draw does (:func:`_check_weights`).

    Raises:
        ValueError: A name of a layer off by default, or a
            ``family_weights`` entry of a denied layer. The message names it,
            its layer and the fix.
    """
    denied = resolved["layers"]["deny"]
    off = layers_off_by_default()
    family_layer = {name: family["layer"]
                    for name, family in load_spec()["families"].items()}
    technique_layer = {fn.__name__: fn.layer for fn in all_techniques()}
    named = [
        *(("families.allow", name, family_layer.get(name))
          for name in resolved["families"]["allow"] or ()),
        *(("techniques.allow", name, technique_layer.get(name))
          for name in resolved["techniques"]["allow"] or ()),
        *(("family_weights", name, family_layer.get(name))
          for name in resolved["family_weights"]),
    ]
    dropped = []
    for key, name, layer in named:
        if layer not in denied:
            continue
        if layer in off:
            raise ValueError(
                f"{where}.{key} names {name!r}, of the {layer} layer, which "
                f"{_SPEC} has off by default, so the kind would not draw it: "
                f'write "layers": {{"allow": ["{layer}"]}} in the kind to turn '
                f"the layer on, or take it out of {key}.")
        if key == "family_weights":
            raise ValueError(
                f"{where}.family_weights names {name!r}, of the {layer} layer, "
                f"which {where}.layers.deny names; take it out of "
                "family_weights, or the layer out of layers.deny.")
        dropped.append(f"{name} ({layer})")
    if dropped:
        logger.warning(
            "%s: the allow lists name %s, of a layer layers.deny names, so the "
            "kind does not draw them.", where, ", ".join(dropped))


def _switch_off_translation(where: str, resolved: dict) -> None:
    """With ``translation`` false, leave out of ``resolved`` the translation,
    dialect and translationese families (all run on the translation model) and
    the techniques its allow lists name and their ``family_weights`` entries,
    and log a WARNING naming them.

    An allow list left empty becomes ``None`` beside the other, which still
    names something, so the pool draws what that one names.

    Raises:
        ValueError: ``translation`` is not a bool, or the allow lists name
            nothing but translation and dialect.
    """
    if type(resolved["translation"]) is not bool:
        raise ValueError(
            f"{where}.translation must be true or false, got "
            f"{resolved['translation']!r}.")
    if resolved["translation"]:
        return
    gated_families = (TRANSLATION, DIALECT, TRANSLATIONESE)
    gated = {TRANSLATION, DIALECT, TRANSLATIONESE,
             *languages(), *dialects(), *translationese_names()}
    lists = [resolved["families"], resolved["techniques"]]
    named = [name for given in lists for name in given["allow"] or ()
             if name in gated]
    weighted = [f"family_weights.{family}" for family in gated_families
                if family in resolved["family_weights"]]
    if not named and not weighted:
        return
    for given in lists:
        if given["allow"] is not None:
            given["allow"] = [name for name in given["allow"]
                              if name not in gated] or None
    if named and all(given["allow"] is None for given in lists):
        raise ValueError(
            f"{where} allows only translation, dialect or translationese "
            f'({named}) and {where}.translation is false; set "translation": '
            "true, or allow something else.")
    resolved["family_weights"] = {
        family: weight for family, weight in resolved["family_weights"].items()
        if family not in gated_families}
    logger.warning(
        "%s.translation is false, so the translation, dialect or translationese "
        "it names is left out: %s.", where, named + weighted)


def _check_count(where: str, key: str, value, *, least: int) -> None:
    if type(value) is not int or value < least:
        raise ValueError(
            f"{where}.{key} must be a whole number of at least {least}, got {value!r}.")


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _check_weights(where: str, resolved: dict) -> None:
    """``family_weights``: numbers above 0, for families the kind's pool draws."""
    weights = resolved["family_weights"]
    if not isinstance(weights, dict):
        raise ValueError(
            f"{where}.family_weights must be an object of family: weight, got "
            f"{weights!r}.")
    drawn = sorted({(getattr(fn, "families", None) or [""])[0]
                    for fn in technique_pool(resolved)} - {""})
    for family, weight in weights.items():
        if family not in drawn:
            raise ValueError(
                f"{where}.family_weights names {family!r}, which the kind does "
                f"not draw from; write one of {drawn}.")
        if not _is_number(weight) or weight <= 0:
            raise ValueError(
                f"{where}.family_weights[{family!r}] must be a number above 0, got "
                f"{weight!r}.")


def _check_rates(where: str, rates) -> None:
    if not isinstance(rates, dict):
        raise ValueError(
            f"{where} must be an object of layer: rate, got {rates!r}; write rates "
            f"for {list(_RATE_KEYS)}, or {{}} for the spec's.")
    for layer, rate in rates.items():
        if layer not in _RATE_KEYS:
            raise ValueError(
                f"{where} names {layer!r}; write rates for {list(_RATE_KEYS)}.")
        if not _is_number(rate) or not 0 <= rate <= 1:
            raise ValueError(
                f"{where}[{layer!r}] must be a number from 0 to 1, got {rate!r}.")


def _check_budget(where: str, settings: dict) -> None:
    """The budget keys an adversarial kind or one of its rounds gives.

    ``budget`` is ``"unit"`` (each unit spends up to ``max_complexity``,
    the default), ``"group"`` (the group budget, see
    :func:`redact.jailbreak.plan.assign_combination`) or ``None`` (the
    default); ``max_unit_complexity`` is given only with ``"group"``, at or
    above ``max_complexity``.
    """
    for key in ("max_complexity", "max_obfuscations"):
        if key in settings:
            _check_count(where, key, settings[key], least=0)
    if "exact_techniques" in settings:
        _check_count(where, "exact_techniques", settings["exact_techniques"], least=1)
    if "layer_rates" in settings:
        _check_rates(f"{where}.layer_rates", settings["layer_rates"])
    if ("allow_request_obfuscation" in settings
            and type(settings["allow_request_obfuscation"]) is not bool):
        raise ValueError(
            f"{where}.allow_request_obfuscation must be true or false, got "
            f"{settings['allow_request_obfuscation']!r}.")
    budget = settings.get("budget")
    if budget not in (None, "unit", "group"):
        raise ValueError(
            f'{where}.budget must be "unit", "group" or null, got {budget!r}.')
    ceiling = settings.get("max_unit_complexity")
    if ceiling is not None:
        if budget != "group":
            raise ValueError(
                f'{where}.max_unit_complexity applies only with "budget": '
                '"group"; set that, or take max_unit_complexity out.')
        _check_count(where, "max_unit_complexity", ceiling,
                     least=settings.get("max_complexity", 0))


def _warn_rewrite_rate(where: str, resolved: dict) -> None:
    """Log a WARNING when an adversarial kind's rewrite rate and its layer
    disagree, which leaves the layer undrawn without a word.

    With the layer off, a ``layer_rates.rewrite`` above 0, the kind's or a
    round's, draws nothing. With the layer drawn (the kind's pool holds a
    rewrite technique), a budget round whose rate for it is 0 draws none
    there: the spec's rate is 0, and a round's own ``layer_rates`` replace
    the kind's (``"escalation"`` carries the kind's, :func:`rounds_of`). A
    count round draws the layer whatever the rate.
    """
    rates = [resolved["layer_rates"]]
    if isinstance(resolved["rounds"], list):
        rates += [r.get("layer_rates", {}) for r in resolved["rounds"]]
    if _REWRITE in resolved["layers"]["deny"]:
        if any(given.get(_REWRITE, 0) > 0 for given in rates):
            logger.warning(
                '%s gives the rewrite layer a rate in layer_rates, but the layer '
                'is off, so the kind draws no rewrite: write "layers": {"allow": '
                '["rewrite"]} to turn it on, or take the rate out.', where)
        return
    rounds = rounds_of(resolved)
    if "include_rewrite" not in rounds[0]:
        return
    spec_rate = load_spec().get("layer_rates", {}).get(_REWRITE, 0)
    flat = [i for i, r in enumerate(rounds) if "exact_techniques" not in r
            and r.get("sampling_probs", resolved["layer_rates"]).get(
                _REWRITE, spec_rate) == 0]
    if flat:
        logger.warning(
            "%s draws the rewrite layer, but its budget round(s) %s (from 0) "
            "draw it at rate 0, the spec's rate unless layer_rates give one (a "
            "round's own layer_rates replace the kind's): give "
            "layer_rates.rewrite a rate above 0, in the kind and in each round "
            "that gives its own layer_rates.", where, flat)


def _rounds(where: str, rounds) -> list:
    """The rounds ``rounds`` names, each checked."""
    if rounds == "escalation":
        return default_escalation_schedule()
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(
            f"{where}.rounds must be null, \"escalation\" or a non-empty list of "
            f"rounds, got {rounds!r}.")
    for i, r in enumerate(rounds):
        at = f"{where}.rounds[{i}]"
        if not isinstance(r, dict):
            raise ValueError(f"{at} must be an object, got {r!r}.")
        unknown = [key for key in r if key not in _ROUND_KEYS]
        if unknown:
            raise ValueError(
                f"{at} has unknown key(s) {unknown}; write only {list(_ROUND_KEYS)}.")
        _check_budget(at, r)
    return rounds
