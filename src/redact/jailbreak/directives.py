"""Template files under ``configs/jailbreak/``, the variant pools they hold,
and one variant of a pool applied to a prompt.

A pure template technique draws its text from a *variant pool*: one
directive's variants in a template file. :func:`read_pools` reads a file of
any of these shapes into ``{directive: [variant, ...]}``, each variant a dict
with a ``name``, its text and optionally a ``weight``:

- ``{directive: [{"name", "template"}, ...]}``: framing, the answer formats,
  the answer languages (one directive, ``templates``), asking, the
  distractors, impersonation, indirect, temporal, the neutral containers,
  small talk and the benign roles, and the decode hints of the encodings (a
  directive per encoding, named after its technique);
- ``{directive: [{"name", "prefix", "suffix"}, ...]}``: continuation, which
  puts the prompt between the two, so they hold no slot;
- ``{technique: {"template", ...}}``: structural's ``to_xml`` and
  ``to_markdown``, one template each, read as a pool of one variant named
  after its entry; the entry's other keys (``policy_ids``) are its
  technique's own. ``to_json`` still writes its wrapper in
  ``obfuscation/structural.py``; moving it into the file waits for the
  next eval review.

``personas.json`` and ``cognitive_techniques.json`` define personas and
scenarios, not variants: their modules read them with :func:`load_templates`.

A variant's ``weight`` (a number, at least 0) makes its pool drawn by weight;
0 turns the variant off. No packaged variant has one, and an eval-pool
technique's never should: a weight changes the draws of its whole pool.

A technique marks the pool it draws from with :func:`uses_pool`, and
``pool.py`` checks every marked pool when ``redact.jailbreak`` is imported
(:func:`check_pools`). A template holds the request slot ``{prompt}`` exactly
once, unless its pool says it has none (``request_slot=False``): a piece of
text the technique fills with its own slots and sets into another template,
as the persona templates' mechanics are. A slot is written alone, with no
conversion or format spec, so the text it fills is the value as it is.
"""

import json
import math
import random
import re
import string
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .. import paths

_CONFIGS_JAILBREAK_DIR = paths.jailbreak_configs_dir()

#: The keys a variant may have.
_VARIANT_KEYS = {"name", "template", "prefix", "suffix", "weight"}

#: What a variant's name may hold: it is written into a row's info as
#: ``key=<name>``, the keys joined by ``;``.
_NAME = re.compile(r"[A-Za-z0-9_-]+")


def load_templates(*relative_parts: str) -> dict:
    """The JSON file at ``configs/jailbreak/<relative_parts>``, in its own shape.

    ``load_templates("requests", "answer_templates.json")`` reads
    ``configs/jailbreak/requests/answer_templates.json``.
    """
    path = _CONFIGS_JAILBREAK_DIR.joinpath(*relative_parts)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PoolRef:
    """The variant pool a technique draws from: ``directive`` of the template
    ``file`` (a path under ``configs/jailbreak/``), whose templates may name
    the ``slots`` the technique fills. ``info_key`` is the key of the
    technique's info that names the variant it drew: ``variant``, as
    :func:`apply_template_directive` writes it, continuation's ``style``, or
    None for a technique whose info names none (structural's).
    ``request_slot`` is False for a pool whose templates hold no
    ``{prompt}``: the technique sets the filled variant into another
    template, which holds the request. ``optional`` is True for a technique
    that draws from its pool only at a rate (a decode hint), so its info
    names a variant only when it drew one."""

    file: str
    directive: str
    slots: tuple[str, ...] = ("prompt",)
    info_key: str | None = "variant"
    request_slot: bool = True
    optional: bool = False

    @property
    def name(self) -> str:
        """``<file>:<directive>``, the pool's name in errors and the variant floor."""
        return f"{self.file}:{self.directive}"


def uses_pool(
    file: str, directive: str, slots: tuple[str, ...] = ("prompt",),
    info_key: str | None = "variant", request_slot: bool = True,
    optional: bool = False,
) -> Callable[[Callable], Callable]:
    """Mark a technique as drawing from the pool ``directive`` of ``file``,
    filling ``slots`` and naming the variant in its info under ``info_key``;
    ``request_slot=False`` for a pool whose templates hold no ``{prompt}``,
    ``optional=True`` for one the technique draws from only at a rate.
    Sets its ``variant_pool`` (a :class:`PoolRef`)."""
    def mark(fn: Callable) -> Callable:
        fn.variant_pool = PoolRef(
            file, directive, tuple(slots), info_key, request_slot, optional)
        return fn
    return mark


def read_pools(file: str, folder: Path = _CONFIGS_JAILBREAK_DIR) -> dict[str, list]:
    """The variant pools of the template file ``file`` under ``folder``
    (``configs/jailbreak/`` by default; ``file`` such as
    ``"requests/answer_templates.json"``), by directive, whatever the file's
    shape (the module docstring). A key starting with ``_`` is a comment.
    Read as written: :func:`check_pools` checks them."""
    data = json.loads((Path(folder) / file).read_text(encoding="utf-8"))
    pools = {}
    for directive, entry in data.items():
        if directive.startswith("_"):
            continue
        if isinstance(entry, dict) and "template" in entry:
            entry = [{"name": directive, "template": entry["template"]}]
        pools[directive] = entry
    return pools


def enabled(variants: list[dict]) -> list[dict]:
    """The variants of a pool that may be drawn: those of weight above 0."""
    return [variant for variant in variants if variant.get("weight", 1) > 0]


def draw_variant(variants: list[dict], rng: random.Random | None = None) -> dict:
    """One variant of a pool, drawn from ``rng``, else the global ``random``.

    A pool none of whose variants gives a ``weight`` is drawn with
    ``rng.choice``, so a seed draws the variant it drew before weights
    existed; a pool where one does is drawn with ``rng.choices`` by weight, a
    variant without one weighing 1. A variant of weight 0 is never drawn, and
    a pool with one variant left makes no draw, leaving ``rng`` as it was.
    """
    pool = enabled(variants)
    if len(pool) == 1:
        return pool[0]
    rng = rng or random
    if any("weight" in variant for variant in variants):
        return rng.choices(pool, weights=[v.get("weight", 1) for v in pool])[0]
    return rng.choice(pool)


def check_pools(refs: Iterable[PoolRef], folder: Path = _CONFIGS_JAILBREAK_DIR) -> None:
    """Raise unless every pool of ``refs`` is one its techniques can draw.

    A pool is a non-empty list of variants with non-empty names of letters,
    digits, ``_`` and ``-``, each once; a variant has a ``template``, or a
    ``prefix`` and a ``suffix`` (strings), and no key beyond those, ``name``
    and ``weight``. A template names the request slot ``{prompt}`` exactly
    once (unless the pool has no ``request_slot``) and no slot but the
    pool's ``slots``, each with no conversion or format spec; a prefix and a
    suffix are not formatted, so they hold none. A ``weight`` is a number of
    at least 0, and at least one variant is above 0.

    Raises:
        ValueError: The first problem, naming the file, the directive and
            the variant.
    """
    refs = sorted(set(refs), key=lambda ref: ref.name)
    files = {ref.file: read_pools(ref.file, folder) for ref in refs}
    for ref in refs:
        _check_pool(ref, files[ref.file])


def _check_pool(ref: PoolRef, pools: dict[str, list]) -> None:
    """Raise unless ``ref``'s pool, among the ``pools`` of its file, is one
    its technique can draw."""
    where = f"configs/jailbreak/{ref.file}, directive {ref.directive!r}"
    if ref.directive not in pools:
        raise ValueError(f"{where}: the file has no such directive.")
    variants = pools[ref.directive]
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{where}: not a non-empty list of variants.")
    seen = set()
    for variant in variants:
        name = variant.get("name") if isinstance(variant, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{where}: a variant has no name; give it a non-empty one.")
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"{where}, variant {name!r}: the name holds a character other "
                "than a letter, a digit, _ or -; a name is written into a row's "
                "info as key=<name>, where ; and = separate the parts.")
        if name in seen:
            raise ValueError(f"{where}, variant {name!r}: the name is taken; "
                             "give each variant of a directive its own.")
        seen.add(name)
        _check_variant(f"{where}, variant {name!r}", variant, ref.slots,
                       ref.request_slot)
    if not enabled(variants):
        raise ValueError(f"{where}: every variant has weight 0; give one a "
                         "weight above 0, or leave the weights out.")


def _check_variant(
    where: str, variant: dict, slots: tuple[str, ...], request_slot: bool = True,
) -> None:
    """Raise unless ``variant`` has a text its technique can fill and a
    sound weight; its template holds ``{prompt}`` when the pool has a
    ``request_slot``."""
    if unknown := sorted(set(variant) - _VARIANT_KEYS):
        raise ValueError(f"{where}: unknown keys {unknown}; a variant has "
                         f"{sorted(_VARIANT_KEYS)}.")
    weight = variant.get("weight", 1)
    if (isinstance(weight, bool) or not isinstance(weight, int | float)
            or not math.isfinite(weight) or weight < 0):
        raise ValueError(f"{where}: weight must be a number of at least 0 "
                         f"(0 turns the variant off), got {weight!r}.")
    if "template" in variant:
        template = variant["template"]
        if not isinstance(template, str):
            raise ValueError(f"{where}: the template is not a string.")
        named = slots_of(where, template)
        if request_slot and "prompt" not in named:
            raise ValueError(f"{where}: the template has no {{prompt}} slot.")
        if request_slot and named.count("prompt") > 1:
            raise ValueError(
                f"{where}: the template has the {{prompt}} slot "
                f"{named.count('prompt')} times; it needs exactly one, or the "
                "request is written more than once.")
        if extra := sorted(set(named) - set(slots)):
            raise ValueError(
                f"{where}: the template names {extra}, which its technique "
                f"does not fill (it fills {list(slots)}); write a literal "
                "brace as {{ or }}.")
    elif not (isinstance(variant.get("prefix"), str)
              and isinstance(variant.get("suffix"), str)):
        raise ValueError(
            f"{where}: a variant has a template, or a prefix and a suffix (strings).")


def slots_of(where: str, template: str) -> list[str]:
    """The slots ``template`` names, as ``str.format_map`` reads them, in
    order and once for each time it names one.

    Raises:
        ValueError: The template does not parse, or a slot has a conversion
            or a format spec (``{prompt!r}``, ``{prompt:>9}``, which would
            write something other than the value); the message begins with
            ``where``.
    """
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{where}: the template does not parse ({exc}); write "
                         "a literal brace as {{ or }}.") from None
    slots = []
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if spec or conversion:
            written = (f"!{conversion}" if conversion else "") + (
                f":{spec}" if spec else "")
            raise ValueError(
                f"{where}: the slot {{{field}}} has a conversion or a format spec "
                f"({written}); write it as {{{field}}}, so the text it fills is "
                "the value as it is.")
        slots.append(field)
    return slots


def apply_template_directive(
    templates: dict[str, list[dict]],
    directive: str,
    prompt: str,
    info_key: str = "directive",
    extra: dict[str, str] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Apply one variant of ``directive`` to ``prompt``.

    ``templates[directive]`` is a pool of ``{"name": ..., "template": ...}``
    variants (:func:`read_pools`), whose slots are ``{prompt}`` and the keys
    of ``extra``. The variant is drawn by :func:`draw_variant` from ``rng``,
    else from the global ``random``. Returns the text and
    ``"<info_key>=<directive>;variant=<name>"``.
    """
    variant = draw_variant(templates[directive], rng)
    result = variant["template"].format_map({**(extra or {}), "prompt": prompt})
    return result, f"{info_key}={directive};variant={variant['name']}"
