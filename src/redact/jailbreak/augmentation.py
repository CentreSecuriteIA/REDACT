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
the natural kind draws only what the spec marks natural.
"""

import copy
import json
import logging
from functools import lru_cache
from numbers import Real
from pathlib import Path

from redact.config_overlay import config_file
from redact.types import ALL_ENTRY_TYPES

from .plan import default_escalation_schedule
from .pool import TRANSLATION, all_techniques, languages, technique_pool
from .technique import DEFAULT_KINDS, KINDS, family_kinds, load_spec

logger = logging.getLogger(__name__)

_COMMON_KEYS = ("entry_types", "translation", "layers", "families", "techniques",
                "pure_only", "family_weights", "iterations")
_KIND_KEYS = {
    "adversarial": (*_COMMON_KEYS, "rounds", "max_complexity", "max_obfuscations",
                    "layer_rates", "allow_request_obfuscation"),
    "natural": (*_COMMON_KEYS, "exact_techniques"),
}
_ROUND_KEYS = ("exact_techniques", "max_complexity", "max_obfuscations",
               "layer_rates", "allow_request_obfuscation")
_RATE_KEYS = ("hacking", "manipulation", "requests", "request_obfuscation")
_LAYERS = ("hacking", "manipulation", "obfuscation", "requests")
_SPEC = "combination_spec.json"

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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _section_file(config_dir: str | Path | None) -> dict:
    """``augmentations.json``, read again when the file changes."""
    path = config_file("jailbreak/augmentations.json", config_dir)
    return _read(str(path), path.stat().st_mtime_ns)


def defaults(config_dir: str | Path | None = None) -> dict:
    """The default section, both kinds, as a copy the caller may change: the
    packaged one, or ``config_dir``'s when it has the file."""
    return copy.deepcopy({kind: _section_file(config_dir)[kind] for kind in KINDS})


def natural_only_families(config_dir: str | Path | None = None) -> list[str]:
    """The families the adversarial kind draws only when its
    ``families.allow`` names them."""
    return list(_section_file(config_dir)["natural_only_families"])


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
    replaces the packaged one when it has the file. ``layers``, ``families`` and
    ``techniques`` come back with both their lists (``allow`` ``None`` for
    all), and a kind with ``rounds`` with ``iterations`` set to their number.
    The adversarial ``families.deny`` holds every family of
    :func:`natural_only_families` its ``families.allow`` does not name.
    An allow list names only families and techniques whose spec ``kinds``
    hold the kind (:func:`_check_kinds`).
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
    unknown = [key for key in given if key not in _KIND_KEYS[kind]]
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {unknown}; write only "
            f"{list(_KIND_KEYS[kind])}.")

    resolved = {**copy.deepcopy(default), **copy.deepcopy(given)}
    _check_entry_types(where, resolved["entry_types"])
    for level, keys in (("layers", ("deny",)), ("families", ("allow", "deny")),
                        ("techniques", ("allow", "deny"))):
        if level in given:
            lists = _lists(where, level, given[level], keys)
            resolved[level] = _overlay(
                where, level, default[level],
                {key: lists[key] for key in keys if key in given[level]})
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
    _check_kinds(where, kind, resolved)
    _switch_off_translation(where, resolved)
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
    return resolved


def rounds_of(resolved: dict) -> list[dict]:
    """The sampler settings of each round of a resolved kind, one dict a round.

    A natural round draws ``exact_techniques``. An adversarial round is the
    budget keys, each layer's ``include_*``, and the round's own keys from
    ``rounds`` merged over them (``"escalation"`` is
    :func:`default_escalation_schedule`), a round's ``layer_rates`` passed on
    as the sampler's ``sampling_probs``.
    """
    if "exact_techniques" in resolved:
        return [{"exact_techniques": resolved["exact_techniques"]}
                for _ in range(resolved["iterations"])]
    base = {
        "max_complexity": resolved["max_complexity"],
        "max_obfuscations": resolved["max_obfuscations"],
        **{f"include_{layer}": layer not in resolved["layers"]["deny"]
           for layer in _LAYERS},
    }
    if resolved["allow_request_obfuscation"]:
        base["allow_request_obfuscation"] = True
    rounds = resolved["rounds"]
    if rounds is None:
        return [dict(base) for _ in range(resolved["iterations"])]
    if rounds == "escalation":
        return [{**base, **r} for r in default_escalation_schedule()]
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


def _lists(where: str, level: str, value, keys: tuple[str, ...]) -> dict:
    """``value`` as ``{"allow": list | None, "deny": list}``, the keys of ``keys``."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{where}.{level} must be an object of {list(keys)}, got {value!r}.")
    unknown = [key for key in value if key not in keys]
    if unknown:
        raise ValueError(
            f"{where}.{level} has unknown key(s) {unknown}; write only {list(keys)}.")
    out = {}
    if "allow" in keys:
        allow = value.get("allow")
        if allow is not None and (not isinstance(allow, list) or not allow):
            raise ValueError(
                f"{where}.{level}.allow must be a non-empty list, got {allow!r}; "
                "write null for all, or name what to leave out under deny.")
        out["allow"] = None if allow is None else list(allow)
    deny = value.get("deny", [])
    if not isinstance(deny, list):
        raise ValueError(f"{where}.{level}.deny must be a list, got {deny!r}.")
    out["deny"] = list(deny)
    if "allow" in out and out["allow"]:
        both = [name for name in out["allow"] if name in out["deny"]]
        if both:
            raise ValueError(
                f"{where}.{level} names {both} under both allow and deny; write "
                "each under one of them.")
    return out


def _overlay(where: str, level: str, default: dict, given: dict) -> dict:
    """The lists ``given`` holds over ``default``'s, one by one.

    A name given under one list leaves the default's other list, so a deny
    given alone is the default's allow less those names.
    """
    out = {**default, **given}
    for key, other in (("allow", "deny"), ("deny", "allow")):
        if given.get(key) and other not in given and out.get(other):
            out[other] = [name for name in out[other] if name not in given[key]]
            if other == "allow" and not out[other]:
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
    for name in resolved["layers"]["deny"]:
        if name not in layers:
            raise ValueError(
                f"{where}.layers.deny names {name!r}, which is not a layer of "
                f"{_SPEC}; write one of {layers}.")
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


def _check_kinds(where: str, kind: str, resolved: dict) -> None:
    """Every family and technique the allow lists name is one ``kind`` may
    draw: its spec ``kinds`` hold it, as :func:`technique_pool` reads them.

    Raises:
        ValueError: An allow list names one the spec keeps from ``kind``, as
            a copy of ``augmentations.json`` made before it left the kind
            does. The message names it and both fixes.
    """
    techniques = {fn.__name__: getattr(fn, "kinds", DEFAULT_KINDS)
                  for fn in all_techniques()}
    families = load_spec()["families"]
    for level in ("families", "techniques"):
        for name in resolved[level]["allow"] or ():
            if name not in (families if level == "families" else techniques):
                continue        # _check_names refuses a name nothing has
            kinds = (family_kinds(name) if level == "families"
                     else techniques[name])
            if kind not in kinds:
                raise ValueError(
                    f"{where}.{level}.allow names {name!r}, which the {kind} kind "
                    f"does not draw: its kinds in {_SPEC} are {list(kinds)}. Take it "
                    f"out of {level}.allow (a config_dir copy of augmentations.json "
                    f"made before it left the {kind} kind still names it), or add "
                    f'"{kind}" to its kinds in {_SPEC}.')


def _switch_off_translation(where: str, resolved: dict) -> None:
    """With ``translation`` false, leave out of ``resolved`` the translation
    family and techniques its allow lists name and its ``family_weights``
    entry, and log a WARNING naming them.

    An allow list left empty becomes ``None`` beside the other, which still
    names something, so the pool draws what that one names.

    Raises:
        ValueError: ``translation`` is not a bool, or the allow lists name
            nothing but translation.
    """
    if type(resolved["translation"]) is not bool:
        raise ValueError(
            f"{where}.translation must be true or false, got "
            f"{resolved['translation']!r}.")
    if resolved["translation"]:
        return
    translation = {TRANSLATION, *languages()}
    lists = [resolved["families"], resolved["techniques"]]
    named = [name for given in lists for name in given["allow"] or ()
             if name in translation]
    weighted = TRANSLATION in resolved["family_weights"]
    if not named and not weighted:
        return
    for given in lists:
        if given["allow"] is not None:
            given["allow"] = [name for name in given["allow"]
                              if name not in translation] or None
    if named and all(given["allow"] is None for given in lists):
        raise ValueError(
            f"{where} allows only translation ({named}) and {where}.translation "
            'is false; set "translation": true, or allow something else.')
    resolved["family_weights"] = {
        family: weight for family, weight in resolved["family_weights"].items()
        if family != TRANSLATION}
    logger.warning(
        "%s.translation is false, so the translation it names is left out: %s.",
        where, named + ["family_weights.translation"] * weighted)


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
    """The budget keys an adversarial kind or one of its rounds gives."""
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
