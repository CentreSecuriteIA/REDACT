"""The technique counts the docs quote, against the spec.

``docs/techniques.md`` holds two tables made from the spec, each between a
``<!-- BEGIN <name> ... -->`` and an ``<!-- END <name> -->`` comment: the
active techniques per layer and per family. A technique counts in its first
family only, and a retired one (``deprecated``) not at all. A change to the
spec that moves them fails here until the tables are made again, from the
repository root, with ``PYTHONPATH=src py -m tests.jailbreak.test_docs_counts``.

The few sentences that quote the totals (the jailbreak heading of
``CLAUDE.md``, ``README.md`` and the introduction of ``techniques.md``) are
written by hand and checked against the same counts.
"""

import re
from pathlib import Path

import pytest

from redact.jailbreak.technique import load_spec

_REPO = Path(__file__).parents[2]
_TECHNIQUES_MD = _REPO / "docs" / "techniques.md"


def _by_family() -> dict[str, list[str]]:
    """The active techniques of each family that has one, by their first
    family, in spec order; the families in layer order, then by their order
    within the layer, then by name."""
    spec = load_spec()
    families = spec["families"]
    members: dict[str, list[str]] = {}
    for name, entry in spec["techniques"].items():
        if not name.startswith("_") and "deprecated" not in entry:
            members.setdefault(entry["families"][0], []).append(name)
    layers = spec["layer_order"]
    return {family: members[family] for family in sorted(members, key=lambda f: (
        layers.index(families[f]["layer"]), families[f].get("within_layer_order", 0),
        f))}


def _which(names: list[str], tag: str) -> str:
    """``yes`` when every technique of ``names`` has the spec ``tag``, ``no``
    when none has it, else the names of those that have it."""
    techniques = load_spec()["techniques"]
    having = [name for name in names if techniques[name].get(tag)]
    if len(having) == len(names):
        return "yes"
    return ", ".join(f"`{name}`" for name in having) or "no"


def layer_table() -> str:
    """The families and active techniques of each layer, with its cap."""
    spec = load_spec()
    by_family = _by_family()
    rows = ["| Layer | Families | Techniques | Picks per combination |",
            "|---|---|---|---|"]
    for layer in spec["layer_order"]:
        names = [f for f in by_family if spec["families"][f]["layer"] == layer]
        cap = spec["layer_max_picks"].get(layer)
        rows.append(f"| `{layer}` | {len(names)} | "
                    f"{sum(len(by_family[f]) for f in names)} | "
                    f"{f'at most {cap}' if cap else 'no cap'} |")
    rows.append(f"| all | {len(by_family)} | "
                f"{sum(len(names) for names in by_family.values())} | |")
    return "\n".join(rows)


def family_table() -> str:
    """Each family's layer, dimension, order, active techniques and their
    model calls and benign data."""
    families = load_spec()["families"]
    rows = ["| Layer | Family | Dimension | Order | Count | Calls a model "
            "| Needs benign data | Techniques |",
            "|---|---|---|---|---|---|---|---|"]
    for family, names in _by_family().items():
        entry = families[family]
        rows.append(
            f"| {entry['layer']} | `{family}` | `{entry['dimension']}` "
            f"| {entry.get('within_layer_order') or ''} | {len(names)} "
            f"| {_which(names, 'requires_llm')} | {_which(names, 'requires_benign')} "
            f"| {', '.join(f'`{name}`' for name in names)} |")
    return "\n".join(rows)


_TABLES = {"layers": layer_table, "families": family_table}


def _block(name: str) -> re.Pattern:
    """The table ``name`` between its marker comments, the markers kept in
    groups 1 and 3."""
    return re.compile(rf"(<!-- BEGIN {name}\b[^>]*-->\n)(.*?)(\n<!-- END {name} -->)",
                      re.S)


def regenerate(text: str) -> str:
    """``text`` with each generated table made again from the spec."""
    for name, table in _TABLES.items():
        text = _block(name).sub(lambda m, table=table: m[1] + table() + m[3], text)
    return text


@pytest.mark.parametrize("name", sorted(_TABLES))
def test_a_generated_table_of_techniques_md_is_the_spec_s(name):
    match = _block(name).search(_TECHNIQUES_MD.read_text(encoding="utf-8"))
    assert match, f"docs/techniques.md has no {name} table between its markers"
    assert match[2] == _TABLES[name]()


def _counts() -> dict[str, int]:
    """The totals the docs quote: active, registered and retired
    techniques, the spec's layers (the rewrite layer, off by default, among
    them) and the families that hold an active technique."""
    spec = load_spec()
    by_family = _by_family()
    registered = sum(not name.startswith("_") for name in spec["techniques"])
    active = sum(len(names) for names in by_family.values())
    return {"active": active, "registered": registered, "retired": registered - active,
            "layers": len(spec["layer_order"]), "families": len(by_family)}


_SENTENCES = [
    ("CLAUDE.md", r"\*\*(?P<active>\d+) active single-turn techniques\*\* "
                  r"\((?P<registered>\d+) registered, (?P<retired>\d+) retired\) in "
                  r"\*\*(?P<layers>\d+) layers → (?P<families>\d+) families\*\*"),
    ("README.md", r"variations of the inputs, drawn from (?P<active>\d+) "
                  r"techniques in (?P<families>\d+) families"),
    ("docs/techniques.md", r"draws from (?P<active>\d+) active techniques in "
                           r"(?P<layers>\d+) layers and (?P<families>\d+) families"),
]


@pytest.mark.parametrize(("path", "pattern"), _SENTENCES)
def test_a_sentence_that_quotes_the_totals_quotes_the_spec_s(path, pattern):
    text = re.sub(r"\s+", " ", (_REPO / path).read_text(encoding="utf-8"))
    match = re.search(pattern, text)
    assert match, f"{path} no longer has this sentence; update it or the pattern"
    quoted = {key: int(value) for key, value in match.groupdict().items()}
    assert quoted == {key: _counts()[key] for key in quoted}


if __name__ == "__main__":
    _TECHNIQUES_MD.write_text(
        regenerate(_TECHNIQUES_MD.read_text(encoding="utf-8")), encoding="utf-8")
