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

#: The keys of a generator entry that say where its rows come from and how
#: their techniques are named; every other key goes into each entry it makes.
_GENERATOR_KEYS = ("file", "where", "name")
#: The keys a data row may give its technique's entry, over the generator's.
_ROW_KEYS = ("complexity", "since", "deprecated")

DISCARDED = "DISCARDED"

#: The kinds of the augmentation section; a spec ``kinds`` names some of them.
KINDS = ("adversarial", "natural")
#: The kinds of a family the spec gives no ``kinds``, and of its techniques.
DEFAULT_KINDS = ("adversarial",)

#: What a spec ``check`` may name: how a technique's model reply is checked.
#: ``typos``, the typo check; ``scenario``, the scenario check; ``word_list``,
#: the strict parse of a marked-word list; ``translate``, translation's own
#: check and retry. The spec only declares it; the technique runs the check.
CHECKS = ("typos", "scenario", "word_list", "translate")

#: The spec version of a technique whose entry gives no ``since``: the
#: techniques the library had before the spec dated them.
FIRST_VERSION = "1.0"
_VERSION = re.compile(r"[0-9]+(\.[0-9]+)*")

#: The ISO 15924 code of the Latin script. A translation technique's
#: ``script`` names the script it writes (``obfuscation/translation.py``);
#: a technique tagged ``needs_latin`` changes only Latin text, so the rules
#: keep it from a translation into any other.
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
    over them. A row's technique still needs its function, which the
    family's module builds from the same file. A key starting with ``_`` is
    a comment.

    Raises:
        ValueError: A generator makes a name the spec, or another of its
            rows, has already.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    techniques = dict(spec["techniques"])
    for family, generator in spec.get("generators", {}).items():
        if family.startswith("_"):
            continue
        given = {key: value for key, value in generator.items()
                 if key not in _GENERATOR_KEYS and not key.startswith("_")}
        for row in data_rows(generator["file"], Path(path).parent):
            if "where" in generator and not row[generator["where"]]:
                continue
            name = generator["name"].format(slug=row["slug"])
            if name in techniques:
                raise ValueError(
                    f"combination_spec.json: the generator of {family!r} makes "
                    f"{name!r}, which the spec has already; give its row in "
                    f"{generator['file']} another slug.")
            techniques[name] = {"families": [family], **given,
                                **{key: row[key] for key in _ROW_KEYS if key in row}}
    return {**spec, "techniques": techniques}


def data_rows(file: str, folder: Path = _CONFIGS) -> list[dict]:
    """The rows of the data file ``file`` under ``folder``
    (``configs/jailbreak/`` by default): a JSON object whose ``rows`` list
    holds one row per technique."""
    return json.loads((Path(folder) / file).read_text(encoding="utf-8"))["rows"]


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


def _checked_version(whose: str, key: str, version) -> str:
    """``version``, checked to be dotted whole numbers, as :func:`version_key`
    reads them."""
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(
            f"combination_spec.json: the {key} of {whose} must be a version of "
            f'dotted whole numbers, such as "1.1", got {version!r}.')
    return version


def tag(funcs: list[Callable]) -> None:
    """Set each function's spec tags as attributes of the function.

    ``families``, ``complexity`` (0-4), ``requires_llm``, ``requires_benign``,
    ``encode_weight`` (``"light"``, ``"heavy"`` or None), ``needs_latin``
    (whether it changes only Latin text, so a translation into another
    script would leave it nothing to do; False when absent), ``max_tokens``
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
        ValueError: A technique's ``max_tokens`` is not a whole number of at
            least 1, its ``needs_latin`` not true or false, its ``kinds``
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
        fn.families = t.get("families", [])
        fn.complexity = t.get("complexity", 1)
        fn.requires_llm = t.get("requires_llm", False)
        # Needs pre-generated benign Q&A data (FSH/DAP). Distinct from
        # requires_llm: the data is generated once up front, not per call — so
        # these aren't per-sample LLM techniques, but they still can't run in a
        # strictly no-LLM (pure_only) run.
        fn.requires_benign = t.get("requires_benign", False)
        fn.encode_weight = t.get("encode_weight", None)
        needs_latin = t.get("needs_latin", False)
        if not isinstance(needs_latin, bool):
            raise ValueError(
                f"Technique {name!r}: needs_latin must be true or false, got "
                f"{needs_latin!r}.")
        fn.needs_latin = needs_latin
        fn.cross_incompatible_with = t.get("cross_incompatible_with", [])
        # Reply limit for the technique's LLM calls; None keeps the model's
        # default. chain.run_chain puts it on each request.
        limit = t.get("max_tokens")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
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
