"""What a technique is: a transform or a generator of model calls, its result,
and the tags the spec gives it.

A transform is ``(prompt) -> (text, info)``. A generator yields
:class:`~redact.llm_pipeline.LLMRequest` objects, is resumed with each reply,
and returns ``(text, info)``. ``info`` is ``""`` or ``key=value;...``; a
rejection's starts with :data:`DISCARDED`. A technique that makes a random
choice draws it from the ``rng`` keyword (a ``random.Random``), else from the
global ``random``.

A technique may carry an applicability (:func:`applicable_when`): the plan
draws it only for an input it is true for (:func:`applies`).

A family whose techniques come one per row of a data file (the languages of
``languages.json``) is a ``generators`` entry of the spec, which
:func:`load_spec` expands into technique entries, so every reader sees one
``techniques`` dict.
"""

import json
import re
from collections.abc import Callable, Generator
from functools import lru_cache
from pathlib import Path

from redact import paths
from redact.llm_pipeline import LLMRequest

_CONFIGS = paths.jailbreak_configs_dir()
_SPEC_PATH = _CONFIGS / "combination_spec.json"
_SPEC = "combination_spec.json"

#: The keys of a generator entry that say where its rows come from and how
#: their techniques are named; every other key goes into each entry it makes.
_GENERATOR_KEYS = ("file", "where", "name")
#: The keys a data row may give its technique's entry, over the generator's.
#: A row's ``<family>_since`` (``translation_since``) is read over its
#: ``since`` for that family's technique alone.
_ROW_KEYS = ("complexity", "since", "deprecated")
#: The keys a technique entry may give (:func:`tag` reads them); a key
#: starting with ``_`` is a comment.
TECHNIQUE_KEYS = ("families", "complexity", "requires_llm", "requires_benign",
                  "encode_weight", "needs_latin", "max_tokens",
                  "cross_incompatible_with", "kinds", "check", "calls", "since",
                  "deprecated")
#: The keys a family entry may give.
FAMILY_KEYS = ("layer", "dimension", "within_layer_order", "cross_incompatible_with",
               "kinds", "check", "calls", "description")
#: What an ``encode_weight`` may be.
ENCODE_WEIGHTS = ("light", "heavy", None)
#: A technique's complexity, the units of a round's budget.
MAX_COMPLEXITY = 3
#: A spec ``layer_rates`` key that is not a layer: the request obfuscation's.
_REQUEST_OBFUSCATION = "request_obfuscation"

DISCARDED = "DISCARDED"

#: The kinds of the augmentation section; a spec ``kinds`` names some of them.
KINDS = ("adversarial", "natural")
#: The kinds of a family the spec gives no ``kinds``, and of its techniques.
DEFAULT_KINDS = ("adversarial",)

#: What a spec ``check`` may name: how a technique's model reply is checked.
#: ``typos``, the typo check; ``scenario``, the scenario check; ``word_list``,
#: the strict parse of a marked-word list; ``translate``, translation's own
#: check and retry; ``preserve``, the check that a rewrite is the prompt's own
#: request. The spec only declares it; the technique runs the check.
CHECKS = ("typos", "scenario", "word_list", "translate", "preserve")

#: The spec version of a technique whose entry gives no ``since``: the
#: techniques the library had before the spec dated them.
FIRST_VERSION = "1.0"
_VERSION = re.compile(r"[0-9]+(\.[0-9]+)*")

#: The ISO 15924 code of the Latin script. A translation technique's
#: ``script`` names the script it writes (``obfuscation/translation.py``);
#: a technique tagged ``needs_latin`` changes only Latin text. No rule reads
#: either: the family rules keep every such technique from every translation
#: (``translation`` refuses ``encode`` and ``noise``), and a test asserts it.
LATIN = "Latn"

TechniqueResult = tuple[str, str]
TechniqueGen = Generator[LLMRequest, str, TechniqueResult]


def is_discarded(info: str) -> bool:
    """Whether ``info`` is a rejected technique's."""
    return info.startswith(DISCARDED)


def applicable_when(predicate: Callable[[str, str], bool]) -> Callable:
    """Give a technique an applicability: ``predicate(text, entry_type)``,
    true when the technique may be drawn for an input of that text and
    entry type.

    The plan leaves the technique out of the pool of an input it is false
    for (:func:`applies`), so it draws no step that would change nothing or
    does not fit the input. Write ``predicate`` by calling the helper the
    technique itself calls, so the two cannot drift apart. The technique is
    returned as it was, with ``applicable`` set.
    """
    def decorate(fn: Callable) -> Callable:
        fn.applicable = predicate
        return fn
    return decorate


def applies(fn: Callable, text: str, entry_type: str) -> bool:
    """Whether ``fn`` may be drawn for an input of ``text`` and
    ``entry_type``: its ``applicable`` predicate's answer, true for a
    technique without one."""
    predicate = getattr(fn, "applicable", None)
    return predicate is None or bool(predicate(text, entry_type))


@lru_cache(maxsize=1)
def load_spec() -> dict:
    """The combination spec: families, techniques, layer order and rules,
    the techniques of its generators among its techniques (:func:`read_spec`)."""
    return read_spec(_SPEC_PATH)


def read_spec(path: Path) -> dict:
    """The spec at ``path``, each of its ``generators`` expanded into
    technique entries, which follow the ones the spec writes out.

    A generator is keyed by its family's name. Its ``file`` is a data file
    in the folder of ``path`` (:func:`data_rows`); with ``where``, only the
    rows whose column of that name is true count. Each row makes one
    technique, named ``name`` with ``{slug}`` the row's ``slug``, whose
    entry is ``families`` (the generator's family) and the generator's
    other keys, with the row's ``complexity``, ``since`` and ``deprecated``
    over them; the row's ``<family>_since`` (``translation_since``) is that
    family's ``since``, for a language one of whose families came in a later
    spec version than its row. A row's technique still needs its function,
    which the family's module builds from the same file. A key starting
    with ``_`` is a comment. ``layers_off_by_default`` (absent is none)
    lists layers of ``layer_order`` that a kind of the augmentation section
    denies unless
    its ``layers.allow`` names them
    (:func:`redact.jailbreak.augmentation.resolve`). ``decode_hint_rate``
    (absent is 0) is the share of hinted encodings written with a hint
    (:func:`decode_hint_rate`).

    What :func:`tag` does not read off a technique is checked here: each
    family's keys (:data:`FAMILY_KEYS`) and ``layer``, the spec's
    ``layer_rates``, each generator's family and each data row
    (:func:`_checked_row`), so a bad spec fails at import.

    Raises:
        ValueError: A generator makes a name the spec, or another of its
            rows, has already, or is keyed by no family of the spec; a row
            is malformed; a family has an unknown key or a layer that is
            not in ``layer_order``; ``layers_off_by_default`` is not a list
            of layers of ``layer_order``; a ``layer_rates`` entry is not a
            number from 0 to 1 for a layer or ``request_obfuscation``; or
            ``decode_hint_rate`` is not a number from 0 to 1.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    layers, off = spec["layer_order"], spec.get("layers_off_by_default", [])
    if not isinstance(off, list) or any(name not in layers for name in off):
        raise ValueError(
            f"combination_spec.json: layers_off_by_default must be a list of "
            f"layers of layer_order, got {off!r}; write names of {layers}.")
    rate = spec.get("decode_hint_rate", 0)
    if not _is_rate(rate):
        raise ValueError(
            f"combination_spec.json: decode_hint_rate must be a number from 0 "
            f"to 1, the share of hinted encodings written with a hint, got "
            f"{rate!r}.")
    _check_families(spec["families"], layers)
    for key, rate in spec.get("layer_rates", {}).items():
        if not key.startswith("_") and (
                key not in [*layers, _REQUEST_OBFUSCATION] or not _is_rate(rate)):
            raise ValueError(
                f"{_SPEC}: layer_rates[{key!r}] must be a number from 0 to 1 for a "
                f"layer of layer_order or {_REQUEST_OBFUSCATION}, got {rate!r}.")
    techniques = dict(spec["techniques"])
    generators = {family: generator
                  for family, generator in spec.get("generators", {}).items()
                  if not family.startswith("_")}
    readers: dict[str, list[str]] = {}      # a data file: the families it makes
    for family, generator in generators.items():
        if family not in spec["families"]:
            raise ValueError(
                f"{_SPEC}: generators has {family!r}, which is not a family; key "
                "each generator by the family its techniques belong to.")
        readers.setdefault(generator["file"], []).append(family)
    for family, generator in generators.items():
        given = {key: value for key, value in generator.items()
                 if key not in _GENERATOR_KEYS and not key.startswith("_")}
        for row in data_rows(generator["file"], Path(path).parent):
            _checked_row(row, family, generator, readers[generator["file"]])
            if "where" in generator and not row[generator["where"]]:
                continue
            name = generator["name"].format(slug=row["slug"])
            if name in techniques:
                raise ValueError(
                    f"combination_spec.json: the generator of {family!r} makes "
                    f"{name!r}, which the spec has already; give its row in "
                    f"{generator['file']} another slug.")
            entry = {"families": [family], **given,
                     **{key: row[key] for key in _ROW_KEYS if key in row}}
            if f"{family}_since" in row:
                entry["since"] = row[f"{family}_since"]
            techniques[name] = entry
    return {**spec, "techniques": techniques}


def _is_rate(value) -> bool:
    """Whether ``value`` is a number from 0 to 1 (not a bool)."""
    return (not isinstance(value, bool) and isinstance(value, int | float)
            and 0 <= value <= 1)


def _check_families(families: dict, layers: list) -> None:
    """Raise unless each family gives only :data:`FAMILY_KEYS` (``_`` keys
    are comments) and a ``layer`` of ``layers``: a family with another
    layer would put its techniques where no layer order reads them."""
    for name, family in families.items():
        unknown = [key for key in family
                   if key not in FAMILY_KEYS and not key.startswith("_")]
        if unknown:
            raise ValueError(
                f"{_SPEC}: family {name!r} has unknown key(s) {unknown}; write only "
                f"{list(FAMILY_KEYS)}.")
        if family.get("layer") not in layers:
            raise ValueError(
                f"{_SPEC}: the layer of family {name!r} must be one of layer_order, "
                f"{layers}, got {family.get('layer')!r}.")


def _checked_row(row: dict, family: str, generator: dict,
                 families: list[str]) -> None:
    """Raise unless the data row ``row`` of its generator's file (its slug
    checked by :func:`data_rows`) gives what the generator reads: its
    ``where`` column true or false, a ``complexity`` from 0 to 3, versions
    in ``since``, ``deprecated`` and each ``<family>_since``, whose family
    must be one of ``families``, those whose generators read the file. The
    message names the data file and the row, not the spec."""
    file, slug = generator["file"], row["slug"]
    where = generator.get("where")
    if where is not None and not isinstance(row.get(where), bool):
        raise ValueError(
            f"{file}: row {slug!r} must give {where!r} true or false, the column "
            f"the generator of {family!r} in {_SPEC} reads (its where), got "
            f"{row.get(where)!r}.")
    if "complexity" in row:
        _checked_complexity(f"{file}: the complexity of row {slug!r}",
                            row["complexity"])
    for key, value in row.items():
        if key.endswith("_since") and key.removesuffix("_since") not in families:
            raise ValueError(
                f"{file}: row {slug!r} gives {key!r}, which names no family made "
                f"from this file; write <family>_since for one of {families}.")
        if key in ("since", "deprecated") or key.endswith("_since"):
            _checked_version(f"row {slug!r}", key, value, file=file)


def _checked_complexity(whose: str, complexity) -> int:
    """``complexity``, checked to be a whole number from 0 to
    :data:`MAX_COMPLEXITY`; ``whose`` starts the message."""
    if type(complexity) is not int or not 0 <= complexity <= MAX_COMPLEXITY:
        raise ValueError(
            f"{whose} must be a whole number from 0 to {MAX_COMPLEXITY}, got "
            f"{complexity!r}.")
    return complexity


def decode_hint_rate() -> float:
    """The spec's ``decode_hint_rate``: the probability that a hinted
    encoding (``obfuscation/decode_hints.json``) writes its text with a hint
    to decode it; 0 when the spec gives none."""
    return load_spec().get("decode_hint_rate", 0)


def data_rows(file: str, folder: Path = _CONFIGS) -> list[dict]:
    """The rows of the data file ``file`` under ``folder``
    (``configs/jailbreak/`` by default): a JSON object whose ``rows`` list
    holds one row per technique, each named by its ``slug``.

    Raises:
        ValueError: A row has no ``slug``, or an empty one.
    """
    rows = json.loads((Path(folder) / file).read_text(encoding="utf-8"))["rows"]
    for index, row in enumerate(rows):
        if not isinstance(row.get("slug"), str) or not row["slug"]:
            raise ValueError(
                f"{file}: row {index} has no slug, the name its techniques are "
                "made from; give it one.")
    return rows


def family_kinds(family: str) -> list[str]:
    """The kinds that may draw ``family``'s techniques: its spec ``kinds``,
    else :data:`DEFAULT_KINDS`. A technique's own ``kinds`` replace them.

    Raises:
        ValueError: The spec's ``kinds`` is not a list of kinds of :data:`KINDS`.
    """
    entry = load_spec()["families"][family]
    return _checked_kinds(f"family {family!r}", entry.get("kinds", DEFAULT_KINDS))


def _checked_kinds(whose: str, kinds) -> list[str]:
    """``kinds`` as a list, checked to be a non-empty list of :data:`KINDS`."""
    if (not isinstance(kinds, list | tuple) or not kinds
            or any(kind not in KINDS for kind in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError(
            f"combination_spec.json: the kinds of {whose} must be a non-empty list "
            f"of {list(KINDS)}, each once, got {kinds!r}.")
    return list(kinds)


def _checked_check(name: str, check) -> str | None:
    """``check``, checked to be one of :data:`CHECKS` or None."""
    if check is not None and check not in CHECKS:
        raise ValueError(
            f"Technique {name!r}: check must be one of {list(CHECKS)}, got "
            f"{check!r}.")
    return check


def _checked_calls(name: str, calls) -> list[int] | None:
    """``calls`` as ``[low, high]``: a whole number ``n`` is ``[n, n]``; None
    stays None."""
    if calls is None:
        return None
    pair = [calls, calls] if isinstance(calls, int) else calls
    if (not isinstance(pair, list) or len(pair) != 2  # noqa: PLR2004
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in pair)
            or not 1 <= pair[0] <= pair[1]):
        raise ValueError(
            f"Technique {name!r}: calls must be a whole number of at least 1 or "
            f"a [low, high] pair of them, got {calls!r}.")
    return list(pair)


def version_key(version: str) -> tuple[int, ...]:
    """A spec version as the whole numbers it compares by: ``"1.10"`` is
    ``(1, 10)``, so it comes after ``"1.9"``."""
    return tuple(int(part) for part in version.split("."))


def _checked_version(whose: str, key: str, version, file: str = _SPEC) -> str:
    """``version``, checked to be dotted whole numbers, as :func:`version_key`
    reads them; the message names ``file``, the one that gives it."""
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(
            f"{file}: the {key} of {whose} must be a version of "
            f'dotted whole numbers, such as "1.1", got {version!r}.')
    return version


def tag(funcs: list[Callable]) -> None:
    """Set each function's spec tags as attributes of the function.

    ``families``, ``complexity`` (0-3), ``requires_llm``, ``requires_benign``,
    ``encode_weight`` (``"light"``, ``"heavy"`` or None), ``needs_latin``
    (whether it changes only Latin text, which a translation into another
    script would leave it nothing to do on; no rule reads it, a test asserts
    the family rules refuse it beside each such translation; False when
    absent), ``max_tokens``
    (the reply limit of its model calls, None for the model's default) and
    ``cross_incompatible_with`` (the families the technique itself may not
    join, beside its families' own) come from the technique's entry;
    ``layer`` and ``within_layer_order`` from its first family; ``kinds``
    (the augmentation kinds that may draw it) from the technique's entry,
    else its first family's (:func:`family_kinds`). ``check`` (what checks
    its model reply, one of :data:`CHECKS`, None for no check) and ``calls``
    (the model calls one application makes, generation and checks, as
    ``[low, high]``; None when the spec does not say) come the same way,
    from the technique's entry, else its first family's. ``since`` is the
    spec version that added the technique (:data:`FIRST_VERSION` when its
    entry gives none) and ``deprecated`` the one that retired it (None while
    it is active). A function the spec does not name is skipped. Called
    once, when ``redact.jailbreak`` is imported.

    Raises:
        ValueError: A technique's entry has a key not in
            :data:`TECHNIQUE_KEYS`, its ``families`` is not a non-empty list
            of the spec's families, its ``complexity`` not a whole number
            from 0 to 3, its ``requires_llm``, ``requires_benign`` or
            ``needs_latin`` not true or false, its ``encode_weight`` not one
            of :data:`ENCODE_WEIGHTS`, its ``max_tokens`` not a whole number
            of at least 1, its ``kinds``
            (or its family's) not a list of
            :data:`KINDS`, its ``check`` not one of :data:`CHECKS`, its
            ``calls`` not a whole number or a ``[low, high]`` pair of them,
            or its ``since`` or ``deprecated``, or the spec's
            ``version``, not dotted whole numbers. Checked here, so a bad
            spec fails at import instead of as an error row per sample or
            on a resume.
    """
    spec = load_spec()
    families = spec["families"]
    techniques = spec["techniques"]
    _checked_version("the spec", "version", spec.get("version", FIRST_VERSION))

    family_layer = {name: f["layer"] for name, f in families.items()}
    family_wlo = {name: f.get("within_layer_order", 0) for name, f in families.items()}

    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name not in techniques:
            continue
        t = techniques[name]
        unknown = [key for key in t
                   if key not in TECHNIQUE_KEYS and not key.startswith("_")]
        if unknown:
            raise ValueError(
                f"Technique {name!r} has unknown key(s) {unknown} in {_SPEC}; "
                f"write only {list(TECHNIQUE_KEYS)}.")
        fn.families = t.get("families", [])
        if (not isinstance(fn.families, list) or not fn.families
                or any(family not in families for family in fn.families)):
            raise ValueError(
                f"Technique {name!r}: families must be a non-empty list of "
                f"families of {_SPEC}, got {fn.families!r}.")
        fn.complexity = _checked_complexity(
            f"Technique {name!r}: complexity", t.get("complexity", 1))
        for flag in ("requires_llm", "requires_benign", "needs_latin"):
            if not isinstance(t.get(flag, False), bool):
                raise ValueError(
                    f"Technique {name!r}: {flag} must be true or false, got "
                    f"{t[flag]!r}.")
        fn.requires_llm = t.get("requires_llm", False)
        # Needs pre-generated benign Q&A data (FSH/DAP). Distinct from
        # requires_llm: the data is generated once up front, not per call — so
        # these aren't per-sample LLM techniques, but they still can't run in a
        # strictly no-LLM (pure_only) run.
        fn.requires_benign = t.get("requires_benign", False)
        fn.encode_weight = t.get("encode_weight", None)
        if fn.encode_weight not in ENCODE_WEIGHTS:
            raise ValueError(
                f"Technique {name!r}: encode_weight must be \"light\", \"heavy\" "
                f"or absent, got {fn.encode_weight!r}.")
        fn.needs_latin = t.get("needs_latin", False)
        fn.cross_incompatible_with = t.get("cross_incompatible_with", [])
        # Reply limit for the technique's LLM calls; None keeps the model's
        # default. chain.run_chain puts it on each request.
        limit = t.get("max_tokens")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(
                f"Technique {name!r}: max_tokens must be a whole number of at "
                f"least 1, got {limit!r}.")
        fn.max_tokens = limit

        primary_family = fn.families[0] if fn.families else None
        fn.layer = family_layer.get(primary_family, "obfuscation")
        fn.within_layer_order = family_wlo.get(primary_family, 0)
        fn.kinds = (_checked_kinds(f"technique {name!r}", t["kinds"]) if "kinds" in t
                    else family_kinds(primary_family) if primary_family in families
                    else list(DEFAULT_KINDS))
        family = families.get(primary_family, {})
        fn.check = _checked_check(name, t.get("check", family.get("check")))
        fn.calls = _checked_calls(name, t.get("calls", family.get("calls")))
        fn.since = _checked_version(
            f"technique {name!r}", "since", t.get("since", FIRST_VERSION))
        fn.deprecated = (_checked_version(f"technique {name!r}", "deprecated",
                                          t["deprecated"])
                         if "deprecated" in t else None)
