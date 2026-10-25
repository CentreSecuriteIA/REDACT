"""The technique catalog (:mod:`redact.jailbreak.catalog`) against the code.

``docs/catalog.json`` and ``docs/catalog.md`` are made from the code; they
fail here until they are made again, from the repository root, with
``PYTHONPATH=src py -m redact.jailbreak.catalog``. The other tests check
what the catalog says against the registry, the rules, the pools and the
variant pools, each read another way than the catalog reads it, and that
the page lists every technique once and renders its tables whole.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact import paths
from redact.jailbreak import catalog
from redact.jailbreak.augmentation import layers_off_by_default, resolve
from redact.jailbreak.pool import (
    DIALECT,
    MODEL_CHECKS,
    TRANSLATION,
    all_techniques,
    technique_pool,
    variant_pools,
)
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

_REPO = Path(__file__).parents[2]
_DOCS = _REPO / "docs"
_FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="module")
def built() -> dict:
    return catalog.build()


def _active() -> list:
    return [fn for fn in all_techniques() if not fn.deprecated]


# --- the files are current ------------------------------------------------------


def test_the_catalog_json_is_the_code_s(built):
    written = (_DOCS / "catalog.json").read_text(encoding="utf-8")
    assert written == catalog.dumps(built), (
        f"docs/catalog.json is stale: regenerate it with {catalog.REGENERATE}"
    )


def test_the_catalog_is_the_same_under_another_hash_seed(tmp_path, built):
    """Nothing in it follows the order of a set: a build in a process with
    another hash seed writes the same text, through the command line."""
    seed = "54321" if os.environ.get("PYTHONHASHSEED") == "12345" else "12345"
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": os.pathsep.join(
            [str(_REPO / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    done = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "redact.jailbreak.catalog",
            "--out-dir",
            str(tmp_path),
        ],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "catalog.json").read_text(encoding="utf-8") == catalog.dumps(
        built
    )
    assert (tmp_path / "catalog.md").read_text(encoding="utf-8") == catalog.markdown(
        built
    )


def test_check_says_whether_the_files_are_current(tmp_path, built, monkeypatch):
    monkeypatch.setattr(catalog, "build", lambda: built)
    assert catalog.main(["--out-dir", str(tmp_path), "--check"]) == 1
    assert not (tmp_path / "catalog.json").exists()
    assert not (tmp_path / "catalog.md").exists()
    assert catalog.main(["--out-dir", str(tmp_path)]) == 0
    assert catalog.main(["--out-dir", str(tmp_path), "--check"]) == 0


# --- every technique, once ------------------------------------------------------


def test_every_registered_technique_appears_once(built):
    names = [fn.__name__ for fn in all_techniques()]
    assert len(set(names)) == len(names)
    assert list(built["techniques"]) == names
    text = catalog.dumps(built)
    for name in names:
        assert text.count(f'\n  "{name}": {{\n') == 1, name


def test_the_totals_are_the_registry_s(built):
    registered = all_techniques()
    retired = [fn for fn in registered if fn.deprecated]
    head = built["headline"]
    assert head["registered"] == len(registered)
    assert head["active"] == len(registered) - len(retired) == len(_active())
    assert head["retired"] == len(retired)
    assert head["layers"] == len(load_spec()["layer_order"]) == len(built["layers"])
    assert head["families"] == len({fn.families[0] for fn in _active()})
    families = built["families"].values()
    assert sum(len(family["techniques"]) for family in families) == head["active"]
    assert sum(len(family["retired"]) for family in families) == head["retired"]
    assert sum(layer["techniques"] for layer in built["layers"]) == head["active"]
    assert sum(layer["retired"] for layer in built["layers"]) == head["retired"]


def test_retired_techniques_are_flagged(built):
    retired = {fn.__name__: fn.deprecated for fn in all_techniques() if fn.deprecated}
    flagged = {
        name: entry["deprecated"]
        for name, entry in built["techniques"].items()
        if entry["retired"]
    }
    assert flagged == retired
    assert all(
        entry["deprecated"] is None
        for name, entry in built["techniques"].items()
        if name not in retired
    )
    # A retired technique is in no pool and no active technique's partner.
    for name, entry in built["techniques"].items():
        assert not set(entry["refused_with"]) & set(retired), name
        if name in retired:
            assert entry["pools"] == [], name


def test_a_technique_s_tags_are_its_function_s(built):
    for fn in all_techniques():
        entry = built["techniques"][fn.__name__]
        assert (
            entry["layer"],
            entry["families"],
            entry["kinds"],
            entry["complexity"],
            entry["requires_llm"],
            entry["requires_benign"],
            entry["encode_weight"],
            entry["needs_latin"],
            entry["max_tokens"],
            entry["check"],
            entry["calls"],
            entry["since"],
            entry["applicability"],
        ) == (
            fn.layer,
            fn.families,
            fn.kinds,
            fn.complexity,
            fn.requires_llm,
            fn.requires_benign,
            fn.encode_weight,
            fn.needs_latin,
            fn.max_tokens,
            fn.check,
            fn.calls,
            fn.since,
            hasattr(fn, "applicable"),
        ), fn.__name__


# --- how they combine -----------------------------------------------------------


def test_allowed_count_is_a_brute_force_recount(built):
    """Each technique beside each active technique, one compatible() call a
    pair."""
    active = _active()
    for fn in all_techniques():
        entry = built["techniques"][fn.__name__]
        partners = [other for other in active if other is not fn]
        admitted = {other.__name__ for other in partners if compatible([fn], [other])}
        assert entry["allowed_count"] == len(admitted), fn.__name__
        assert set(entry["refused_with"]) == {o.__name__ for o in partners} - admitted
        assert entry["refused_count"] == len(partners) - len(admitted)


def test_refusal_rules_name_a_rule_exactly_when_compatible_refuses():
    registered = all_techniques()
    for a in registered:
        for b in registered:
            if a is not b:
                assert bool(catalog.refusal_rules(a, b)) != bool(
                    compatible([a], [b])
                ), (a.__name__, b.__name__)


def test_each_rule_code_is_where_the_spec_puts_it():
    """The codes, read off the spec directly: a shared family, a capped
    layer, a family's list, a technique's own list, the manipulation
    block's families."""
    spec = load_spec()
    families, caps = spec["families"], spec["layer_max_picks"]
    blocked = set(spec["manipulation_blocked_obfuscation"]["families"])
    registered = all_techniques()
    seen = Counter()
    for a in registered:
        for b in registered:
            if a is b:
                continue
            codes = catalog.refusal_rules(a, b)
            fa, fb = set(a.families), set(b.families)
            cross = {
                g for f in fa for g in families[f].get("cross_incompatible_with", [])
            }
            back = {
                g for f in fb for g in families[f].get("cross_incompatible_with", [])
            }
            expected = {
                "S": bool(fa & fb),
                "L": a.layer == b.layer and caps.get(a.layer) == 1,
                "F": bool(fb & cross or fa & back),
                "T": bool(
                    fb & set(a.cross_incompatible_with)
                    or fa & set(b.cross_incompatible_with)
                ),
            }
            for code, wanted in expected.items():
                assert (code in codes) == wanted, (code, a.__name__, b.__name__)
                seen[code] += wanted
            if a.layer == "manipulation" and fb & blocked:
                assert "M" in codes, (a.__name__, b.__name__)
                seen["M"] += 1
    assert all(seen[code] for code in catalog.RULES), seen


def test_compatibility_binds_both_ways():
    registered = all_techniques()
    for a in registered:
        admitted = {fn.__name__ for fn in compatible([a], registered) if fn is not a}
        for b in registered:
            if b is not a:
                assert (b.__name__ in admitted) == bool(compatible([b], [a])), (
                    a.__name__,
                    b.__name__,
                )


def test_the_family_matrix_is_symmetric(built):
    matrix = built["family_matrix"]
    order = matrix["families"]
    assert order == [
        f
        for f in built["families"]
        if built["families"][f]["techniques"] or built["families"][f]["also_in"]
    ]
    for a in order:
        assert matrix["cells"][a][a][0] == "same"
        for b in order:
            assert matrix["cells"][a][b] == matrix["cells"][b][a], (a, b)


def test_the_family_matrix_counts_the_pairs_of_members(built):
    """Each cell recounted from the techniques' own refusals."""
    techniques = built["techniques"]
    matrix = built["family_matrix"]
    members = {
        family: [
            name
            for name, entry in techniques.items()
            if family in entry["families"] and not entry["retired"]
        ]
        for family in matrix["families"]
    }
    for a in matrix["families"]:
        for b in matrix["families"]:
            if a == b:
                continue
            pairs = [(x, y) for x in members[a] for y in members[b] if x != y]
            ok = sum(y not in techniques[x]["refused_with"] for x, y in pairs)
            status = (
                "always" if ok == len(pairs) else "never" if ok == 0 else "sometimes"
            )
            codes = {
                code
                for x, y in pairs
                for code in techniques[x]["refused_with"].get(y, "")
            }
            assert matrix["cells"][a][b][:3] == [status, ok, len(pairs)], (a, b)
            assert set(matrix["cells"][a][b][3]) == codes, (a, b)


def test_never_with_names_the_families_every_member_of_which_is_refused(built):
    techniques = built["techniques"]
    for name, entry in techniques.items():
        for family, codes in entry["never_with"].items():
            others = [
                n
                for n, e in techniques.items()
                if family in e["families"] and not e["retired"] and n != name
            ]
            assert others and all(n in entry["refused_with"] for n in others), (
                name,
                family,
            )
            assert codes == catalog._union(entry["refused_with"][n] for n in others)
        for family, part in entry["partly_with"].items():
            others = [
                n
                for n, e in techniques.items()
                if family in e["families"] and not e["retired"] and n != name
            ]
            refused = [n for n in others if n in entry["refused_with"]]
            assert (part["refused"], part["members"]) == (len(refused), len(others))
            assert 0 < len(refused) < len(others)


# --- pools ----------------------------------------------------------------------


def _expected_pools() -> dict:
    from redact.recipe import load_dataset_type

    pools = {
        "adversarial": resolve(None, "adversarial"),
        "adversarial_rewrite": resolve(
            {"adversarial": {"layers": {"allow": layers_off_by_default()}}},
            "adversarial",
        ),
        "adversarial_no_translation": resolve(
            {"adversarial": {"translation": False}}, "adversarial"
        ),
        "natural": resolve(None, "natural"),
        "natural_translation": resolve({"natural": {"translation": True}}, "natural"),
    }
    for dataset_type in ("eval", "training"):
        section = load_dataset_type(dataset_type)["augmentations"]
        for kind in ("adversarial", "natural"):
            pools[f"{dataset_type}.{kind}"] = resolve(section, kind)
    return pools


def test_every_pool_listed_is_technique_pool_s(built):
    expected = _expected_pools()
    assert set(built["pools"]) == set(expected)
    for name, resolved in expected.items():
        names = [fn.__name__ for fn in technique_pool(resolved)] if resolved else []
        entry = built["pools"][name]
        assert (entry["techniques"], entry["size"], entry["on"]) == (
            names,
            len(names),
            resolved is not None,
        ), name
        held = {n for n, t in built["techniques"].items() if name in t["pools"]}
        assert held == set(names), name
        if resolved is not None:
            assert built["headline"]["pools"][name] == len(names)


def test_the_pinned_pools_are_the_catalog_s(built):
    pools = built["pools"]
    for name, fixture in (
        ("eval.adversarial", "eval_pool.json"),
        ("natural", "natural_pool.json"),
    ):
        pinned = json.loads((_FIXTURES / fixture).read_text(encoding="utf-8"))
        assert sorted(pools[name]["techniques"]) == pinned


# --- prompts --------------------------------------------------------------------


def test_the_variant_totals_are_the_variant_pools_sums(built):
    pools = variant_pools()
    total = sum(pool["variants"] for pool in pools.values())
    prompts = built["prompts"]
    assert prompts["variants_total"] == built["headline"]["template_variants"] == total
    assert sum(prompts["variants_by_layer"].values()) == total
    assert sum(prompts["variants_by_family"].values()) == total
    by_family = Counter()
    for pool in pools.values():
        by_family[pool["family"]] += pool["variants"]
    assert {f: n for f, n in prompts["variants_by_family"].items() if n} == {
        f: n for f, n in by_family.items() if n
    }
    assert {
        name: pool["variants"] for name, pool in prompts["variant_pools"].items()
    } == {name: pool["variants"] for name, pool in pools.items()}
    for fn in all_techniques():
        ref = getattr(fn, "variant_pool", None)
        assert built["techniques"][fn.__name__]["variants"] == (
            pools[ref.name]["variants"] if ref else 0
        ), fn.__name__


def test_a_technique_loads_a_prompt_exactly_when_it_calls_a_model(built):
    steps = set(built["prompts"]["jailbreak_steps"])
    for name, entry in built["techniques"].items():
        assert bool(entry["prompts"]) == entry["requires_llm"], name
        assert set(entry["prompts"]) <= steps, name
        assert entry["prompt_files"] >= len(entry["prompts"])
    assert built["prompts"]["stage_steps"], "the benign pairs load a prompt"


def test_the_models_a_technique_uses_follow_its_spec(built):
    for name, entry in built["techniques"].items():
        models = set(entry["models"])
        assert ("check" in models) == (entry["check"] in MODEL_CHECKS), name
        # The translation model serves the translations and the dialects.
        assert ("translation" in models) == bool(
            {TRANSLATION, DIALECT} & set(entry["families"])
        ), name
        assert ("benign pairs" in models) == entry["requires_benign"], name
        if entry["requires_llm"]:
            assert models & {"generation", "translation"}, name


def test_a_technique_a_model_checks_loads_a_check_prompt_too(built):
    """A model check is a prompt of its own beside the generation's, loaded
    through a template (``PromptTemplate.load``) as often as directly."""
    for name, entry in built["techniques"].items():
        if entry["check"] in MODEL_CHECKS:
            assert len(entry["prompts"]) >= 2, (name, entry["prompts"])  # noqa: PLR2004


def test_the_prompt_files_are_the_prompt_tree_s(built):
    files = sorted(paths.prompts_dir().rglob("*.json"))
    pipelines = built["prompts"]["pipelines"]
    assert sum(entry["files"] for entry in pipelines.values()) == len(files)
    assert built["headline"]["prompt_files"] == len(files)
    assert Counter({p: e["files"] for p, e in pipelines.items()}) == Counter(
        path.relative_to(paths.prompts_dir()).parts[0] for path in files
    )


# --- the combination examples ---------------------------------------------------


def test_the_combination_examples_count_every_unit(built):
    for kind, sim in built["combinations"].items():
        assert sim["units"] == catalog.SIMULATED_UNITS, kind
        assert sum(sim["by_size"].values()) == sim["units"]
        assert sum(sim["by_layers_touched"].values()) == sim["units"]
        assert sim["techniques_drawn"] + len(sim["never_drawn"]) == sim["pool_size"]
        assert sim["pool_size"] == built["pools"][kind]["size"]
        counts = [n for _, _, n in sim["top_family_pairs"]]
        assert counts == sorted(counts, reverse=True)
        assert len(counts) <= catalog.TOP_FAMILY_PAIRS


# --- the page -------------------------------------------------------------------


@pytest.fixture(scope="module")
def page(built) -> str:
    return catalog.markdown(built)


def test_the_catalog_page_is_the_code_s(page):
    written = (_DOCS / "catalog.md").read_text(encoding="utf-8")
    assert written == page, (
        f"docs/catalog.md is stale: regenerate it with {catalog.REGENERATE}"
    )


def _technique_rows(page: str, name: str) -> list[str]:
    row = re.compile(rf"^\| `{re.escape(name)}`( \(retired [^)]*\))? \|")
    return [line for line in page.splitlines() if row.match(line)]


def test_the_page_lists_every_technique_in_one_row(built, page):
    for fn in all_techniques():
        rows = _technique_rows(page, fn.__name__)
        assert len(rows) == 1, (fn.__name__, len(rows))
        entry = built["techniques"][fn.__name__]
        assert f" | {entry['allowed_count']} | " in rows[0], fn.__name__


def test_the_page_marks_the_retired_techniques(built, page):
    for name, entry in built["techniques"].items():
        marked = f"| `{name}` (retired {entry['deprecated']}) |" in page
        assert marked == entry["retired"], name


def test_the_page_quotes_the_catalog_s_totals(built, page):
    head = built["headline"]
    for label, value in (
        ("Techniques registered", head["registered"]),
        ("Active techniques", head["active"]),
        ("Retired techniques", head["retired"]),
        ("Variant pools", head["variant_pools"]),
        ("Template variants", head["template_variants"]),
    ):
        assert f"| {label} | {value} |" in page, label


def _cells(row: str) -> int:
    return len(re.findall(r"(?<!\\)\|", row)) - 1


def test_every_table_row_has_its_header_s_cells(page):
    """A row with a cell more or less than its header does not render; the
    family matrix is one such table."""
    lines = page.splitlines()
    tables = 0
    for i, line in enumerate(lines):
        if not line.startswith("|---"):
            continue
        tables += 1
        width = line.count("---")
        assert _cells(lines[i - 1]) == width, lines[i - 1][:80]
        for row in lines[i + 1 :]:
            if not row.startswith("|"):
                break
            assert _cells(row) == width, row[:80]
    assert tables


def test_the_family_matrix_has_a_row_and_a_column_per_family(built, page):
    order = built["family_matrix"]["families"]
    section = page.split("\n## Family matrix\n")[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if re.match(r"^\| \d+ \| `", line)]
    assert [_cells(row) for row in rows] == [len(order) + 2] * len(order)


def test_the_page_s_links_point_at_its_headings(page):
    anchors = {
        catalog._anchor(line.lstrip("#").strip())
        for line in page.splitlines()
        if line.startswith("#")
    }
    targets = re.findall(r"\]\(#([^)]+)\)", page)
    assert targets and set(targets) <= anchors


#: The longest the page may grow, in lines.
_MOST_LINES = 4000


def test_the_page_stays_readable_in_length(page):
    assert len(page.splitlines()) < _MOST_LINES
